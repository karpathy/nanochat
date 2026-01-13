"""
Linear layer with FP8 training using static scaling.

All scales (x, w, grad) are set at init time - no runtime computation.
This is the approach used by modded-nanogpt and avoids torch.compile issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# FP8 format constants
FP8_E4M3_MAX = 448.0    # max representable value in float8_e4m3fn
FP8_E5M2_MAX = 57344.0  # max representable value in float8_e5m2

# -----------------------------------------------------------------------------
# Custom FP8 matmul operators (based on modded-nanogpt)
# These use torch.library.custom_op for torch.compile compatibility
#
# All scales are Python floats passed at call time (but set statically at init).


@torch.library.custom_op("nanochat::fp8_mm_static", mutates_args=())
def fp8_mm_op(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 matrix multiply: out = x @ w.T with FP8 quantization.

    Args:
        x: Input activations (2D, contiguous)
        w: Weight matrix (2D, contiguous)
        x_s: Scale for x (x_amax / 448)
        w_s: Scale for w (w_amax / 448)
        grad_s: Scale for gradients in backward (grad_amax / 57344)

    Returns:
        out: Result in bfloat16
        x_f8: x quantized to FP8 (saved for backward)
        w_f8: w quantized to FP8 (saved for backward)
    """
    @torch.compile
    def impl(x: torch.Tensor, w: torch.Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8
    return impl(x, w)


@fp8_mm_op.register_fake
def _(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("nanochat::fp8_mm_static_backward", mutates_args=())
def fp8_mm_backward_op(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for FP8 matmul.

    Args:
        g: Gradient of output (dL/dy)
        x_f8: Saved FP8 activations from forward
        w_f8: Saved FP8 weights from forward
        x_s, w_s, grad_s: Scale factors

    Returns:
        grad_x: Gradient w.r.t. input (bfloat16)
        grad_w: Gradient w.r.t. weights (float32 for optimizer)
    """
    @torch.compile
    def impl(grad: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor):
        grad = grad.contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)

        # grad_x = grad @ W
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )

        # grad_w = x.T @ grad (output in float32 for optimizer stability)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T

        return grad_x, grad_w
    return impl(g, x_f8, w_f8)


@fp8_mm_backward_op.register_fake
def _(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


def _fp8_mm_backward(ctx, grad_out: torch.Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanochat.fp8_mm_static_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None


def _fp8_mm_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


fp8_mm_op.register_autograd(_fp8_mm_backward, setup_context=_fp8_mm_setup_context)

# -----------------------------------------------------------------------------


class LinearFP8(nn.Linear):
    """
    Linear layer with FP8 training using static scaling.

    Scales for x and w are set at initialization - no runtime amax computation.
    Grad scale is computed dynamically from input shape.

    IMPORTANT: This layer assumes it is used as an unembedding/classifier layer
    where the output goes directly into softmax cross-entropy loss. This assumption
    allows us to compute grad_scale dynamically: the gradient of cross-entropy w.r.t.
    logits is (softmax - one_hot), which has element-wise amax of 1. With mean reduction
    over B*T tokens, the gradient amax is 1/(B*T).

    During training, uses FP8 matmul with static x/w scales and dynamic grad scale.
    During inference, uses standard BF16 matmul.

    Args:
        in_features: Input dimension
        out_features: Output dimension (vocabulary size)
        bias: Must be False (FP8 matmul has no bias fusion)
        x_scale: Scale for activations = expected_x_amax / 448. Required.
        w_scale: Scale for weights = expected_w_amax / 448. Required.
        monitor: If True, record actual amax values each forward for get_fp8_stats().
                 Adds small overhead from .item() calls. Default False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        x_scale: float = None,
        w_scale: float = None,
        monitor: bool = False,
    ):
        assert bias is False, "LinearFP8 does not support bias (FP8 matmul has no bias fusion)"
        assert x_scale is not None, "x_scale is required (expected_x_amax / 448)"
        assert w_scale is not None, "w_scale is required (expected_w_amax / 448)"
        super().__init__(in_features, out_features, bias=False)

        self.x_scale = x_scale
        self.w_scale = w_scale
        self.monitor = monitor

        # Observed amax values (updated each forward when monitor=True)
        self._x_amax: float | None = None
        self._w_amax: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # This layer assumes (B, T, C) input for classifier/lm_head
            assert x.ndim == 3, f"Expected input shape (B, T, {self.in_features}), got {x.shape}"
            B, T, C = x.shape
            assert C == self.in_features, f"Expected C={self.in_features}, got {C}"

            # Compute grad_scale dynamically from batch size.
            # Assumption: this is an unembedding layer going into cross-entropy loss.
            # Gradient of CE w.r.t. logits = (softmax - one_hot), element-wise amax = 1.
            # With mean reduction over B*T tokens, grad amax = 1/(B*T).
            grad_amax = 1.0 / (B * T)
            grad_scale = grad_amax / FP8_E5M2_MAX

            # Flatten to 2D, do the matmul, reshape back
            _x = x.flatten(0, -2)  # (B, T, C) -> (B*T, C)
            out, _, _ = torch.ops.nanochat.fp8_mm_static(_x, self.weight, self.x_scale, self.w_scale, grad_scale)
            out = out.reshape(B, T, -1)  # (B*T, V) -> (B, T, V)

            # Record actual amax for monitoring (detect if values exceed static scale assumptions)
            if self.monitor:
                self._x_amax = _x.detach().abs().max().item()
                self._w_amax = self.weight.detach().abs().max().item()
        else:
            # Standard linear forward (inference)
            out = F.linear(x, self.weight.type_as(x))

        return out

    def get_fp8_stats(self) -> dict:
        """Return observed amax values for monitoring.

        Compare these against the expected amax implied by static scales:
        - expected_x_amax = x_scale * 448
        - expected_w_amax = w_scale * 448

        If observed > expected, values are being clipped and you should increase the scale.
        """
        return {
            "x_amax": self._x_amax,
            "w_amax": self._w_amax,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias=False, "
            f"x_scale={self.x_scale:.2e}, w_scale={self.w_scale:.2e}"
        )
