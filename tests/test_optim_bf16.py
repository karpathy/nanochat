"""
Regression tests for mixed-dtype scalar / parameter handling in optim.py.

These cover the MPS Metal Graph compiler crashes seen with
NANOCHAT_DTYPE=bfloat16: scalar hyperparams (fp32) being multiplied with
bf16 params (wte, value_embeds) failed with "mps.multiply requires same
element type". CUDA implicitly promotes mixed-dtype operands; MPS doesn't.
"""
import pytest
import torch

from nanochat.optim import adamw_step_fused


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scalars():
    """0-D fp32 scalar tensors matching what MuonAdamW.__init__ creates."""
    return [
        torch.tensor(1.0, dtype=torch.float32),     # step
        torch.tensor(0.01, dtype=torch.float32),    # lr
        torch.tensor(0.9, dtype=torch.float32),     # beta1
        torch.tensor(0.999, dtype=torch.float32),   # beta2
        torch.tensor(1e-8, dtype=torch.float32),    # eps
        torch.tensor(0.01, dtype=torch.float32),    # wd
    ]


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def _run_adamw(p, grad):
    exp_avg = torch.zeros_like(p)
    exp_avg_sq = torch.zeros_like(p)
    p_before = p.clone()
    adamw_step_fused(p, grad, exp_avg, exp_avg_sq, *_scalars())
    _sync(p.device)
    return p_before


def test_adamw_step_fused_bf16_param_with_fp32_scalars():
    """Regression: adamw_step_fused must not crash when p is bf16 but the
    scalar hyperparams are fp32. This is the standard nanochat config —
    wte and value_embeds are cast to COMPUTE_DTYPE (bf16) to save memory,
    while MuonAdamW's shared scalar tensors remain fp32."""
    device = _device()
    torch.manual_seed(0)
    p = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    grad = torch.randn_like(p)
    p_before = _run_adamw(p, grad)
    assert torch.isfinite(p).all(), "bf16 update produced non-finite values"
    assert not torch.equal(p, p_before), "weight did not change after step"


def test_adamw_step_fused_fp32_param_unchanged():
    """The fp32 path must still work and produce a sensible update —
    the dtype-cast patch should be a no-op when p is already fp32."""
    device = _device()
    torch.manual_seed(0)
    p = torch.randn(64, 32, dtype=torch.float32, device=device)
    grad = torch.randn_like(p)
    p_before = _run_adamw(p, grad)
    assert torch.isfinite(p).all()
    delta = (p - p_before).norm().item()
    assert 0 < delta < 10, f"unreasonable update magnitude: {delta}"
