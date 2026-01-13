"""
Linear layer with FP8 training.

Uses dynamic scaling for activations, weights, and gradients (all computed each forward).

Implementation pattern inspired by torchao.float8:
- Uses @torch._dynamo.allow_in_graph on autograd.Function for torch.compile compatibility
- Scales stay as tensors throughout (no .item() calls)
- Inner @torch.compile on the FP8 matmul kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# FP8 format constants
FP8_E4M3_MAX = 448.0    # max representable value in float8_e4m3fn
FP8_E5M2_MAX = 57344.0  # max representable value in float8_e5m2
EPS = 1e-12             # epsilon for numerical stability in scale computation

# -----------------------------------------------------------------------------
# FP8 kernel functions
# These run the FP8 matmul on H100 tensor cores.
# Note: No @torch.compile here - these are called from within a @torch._dynamo.allow_in_graph
# function, which is already inside the outer torch.compile. Nested compile causes issues.


def _fp8_forward_impl(x: torch.Tensor, w: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor):
    """FP8 forward: out = x @ w.T with FP8 quantization."""
    x_f8 = (x / x_scale).to(torch.float8_e4m3fn)
    w_f8 = (w / w_scale).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        x_f8,
        w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=x_scale,
        scale_b=w_scale,
        use_fast_accum=True,
    )
    return out, x_f8, w_f8


def _fp8_backward_impl(
    grad: torch.Tensor,
    x_f8: torch.Tensor,
    w_f8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    grad_scale: torch.Tensor,
):
    """FP8 backward: compute gradients for x and w."""
    grad = grad.contiguous()
    grad_f8 = (grad / grad_scale).to(torch.float8_e5m2)

    # grad_x = grad @ W
    grad_x = torch._scaled_mm(
        grad_f8,
        w_f8.T.contiguous().T,
        out_dtype=torch.bfloat16,
        scale_a=grad_scale,
        scale_b=w_scale,
        use_fast_accum=False,
    )

    # grad_w = x.T @ grad (output in float32 for optimizer stability)
    grad_w = torch._scaled_mm(
        x_f8.T.contiguous(),
        grad_f8.T.contiguous().T,
        out_dtype=torch.float32,
        scale_a=x_scale,
        scale_b=grad_scale,
        use_fast_accum=False,
    ).T

    return grad_x, grad_w


# -----------------------------------------------------------------------------
# Autograd function with @torch._dynamo.allow_in_graph
# This pattern allows the function to be included in torch.compile graphs.


@torch._dynamo.allow_in_graph
class FP8Matmul(torch.autograd.Function):
    """
    FP8 matrix multiply: out = x @ w.T with dynamic scaling.

    This autograd.Function is decorated with @torch._dynamo.allow_in_graph,
    which tells torch.compile to include it in the compiled graph without
    attempting to trace through it (avoiding issues with .item() calls etc).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, grad_scale: torch.Tensor):
        """
        Args:
            x: Input activations (2D, contiguous) in bfloat16
            w: Weight matrix (2D, contiguous) in bfloat16
            grad_scale: Pre-computed scale for gradients in backward

        Returns:
            out: Result in bfloat16
        """
        # Compute scales dynamically as tensors (no .item()!)
        x_amax = x.abs().max()
        w_amax = w.abs().max()
        x_scale = (x_amax / FP8_E4M3_MAX).clamp(min=EPS).float()
        w_scale = (w_amax / FP8_E4M3_MAX).clamp(min=EPS).float()

        # Run FP8 forward
        out, x_f8, w_f8 = _fp8_forward_impl(x, w, x_scale, w_scale)

        # Save for backward
        ctx.save_for_backward(x_f8, w_f8, x_scale, w_scale, grad_scale)

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_f8, w_f8, x_scale, w_scale, grad_scale = ctx.saved_tensors

        # Run FP8 backward
        grad_x, grad_w = _fp8_backward_impl(grad_out, x_f8, w_f8, x_scale, w_scale, grad_scale)

        return grad_x, grad_w, None  # None for grad_scale


# -----------------------------------------------------------------------------


class LinearFP8(nn.Linear):
    """
    Linear layer with FP8 training.

    During training, uses FP8 matmul with:
    - Dynamic scaling for activations (computed each forward)
    - Dynamic scaling for weights (computed each forward)
    - Dynamic scaling for gradients (computed from input shape)

    During inference, uses standard BF16 matmul.

    IMPORTANT: This layer is currently intended uniquely for use as lm_head (the final projection to vocabulary).
    => It assumes input x has shape (B, T, in_features) where B*T is the batch size
    for cross-entropy loss. grad_scale is computed as (1 / B*T) / FP8_E5M2_MAX, which
    assumes cross-entropy with mean reduction where grad magnitude is ~1/batch_tokens.
    Nothing prevents it from being used as a regular layer except that the grad_scale handling would have to be adjusted.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Must be False for now, might support it later
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        assert bias is False, "LinearFP8 does not support bias (FP8 matmul has no bias fusion)"
        super().__init__(in_features, out_features, bias=False)

        # Latest stats (tensors for logging, detached to avoid grad issues)
        self._x_amax: torch.Tensor | None = None
        self._w_amax: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            assert x.ndim == 3 and x.shape[2] == self.in_features, f"Expected input shape (B, T, {self.in_features}), got {x.shape}"
            B, T, _ = x.shape

            # Flatten to 2D for matmul
            _x = x.flatten(0, -2)  # (B, T, in_features) -> (B*T, in_features)

            # Compute grad_scale as a tensor (assumes cross-entropy with mean reduction)
            grad_scale = torch.tensor(1.0 / (B * T) / FP8_E5M2_MAX, device=x.device, dtype=torch.float32)

            # Run FP8 matmul
            out = FP8Matmul.apply(_x, self.weight, grad_scale)

            # Reshape back
            out = out.reshape(B, T, -1)  # (B*T, out_features) -> (B, T, out_features)

            # Update stats for logging (detach to avoid keeping grad graph)
            self._x_amax = _x.abs().max().detach()
            self._w_amax = self.weight.abs().max().detach()
        else:
            # Standard linear forward (inference)
            out = F.linear(x, self.weight.type_as(x))

        return out

    def get_fp8_stats(self) -> dict:
        """Return the latest FP8 statistics for logging."""
        return {
            "x_amax": self._x_amax.item() if self._x_amax is not None else None,
            "w_amax": self._w_amax.item() if self._w_amax is not None else None,
        }
