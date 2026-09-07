"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

The public API is a single function, fp8_matmul(x, w): a drop-in replacement for
gpt.bf16_matmul that runs the three GEMMs of a Linear layer in FP8. The training
script threads it into the functional forward pass:

    forward(params, x, ..., matmul=fp8_matmul)   # FP8 training
    forward(params, x, ...)                      # BF16 (default), e.g. for evals

This replaces ~2000 lines of torchao (Float8Linear, tensor subclass dispatch,
module tree surgery) with ~150 lines around PyTorch built-ins.

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

Also note the dimension requirements: all matmul dims must be divisible by 16,
and very small matmuls aren't worth it. The model's ve_gate matmul (12 input
channels) stays in bf16 for this reason; the forward pass hardcodes that.

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor bundles FP8
data + scale + metadata and intercepts every aten op via __torch_dispatch__.
We instead use a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass and sees every individual op
    (amax, scale, cast, _scaled_mm) as separate graph nodes that Inductor can fuse
    with surrounding operations.
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 matmul (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

import torch

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


def fp8_matmul(x, w):
    """Drop-in replacement for gpt.bf16_matmul that computes x @ w.T in FP8.

    The weight stays in its original precision (fp32 master weights); only the
    matmuls run in FP8 via the _Float8Matmul autograd function above.
    """
    # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
    # reduced precision input, and we no longer rely on autocast to do this.
    x = x.to(COMPUTE_DTYPE)
    # _scaled_mm only works on 2D tensors, so flatten batch dimensions
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    out_2d = _Float8Matmul.apply(x_2d, w)
    out = out_2d.reshape(*orig_shape[:-1], out_2d.shape[-1])
    return out
