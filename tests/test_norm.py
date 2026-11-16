"""
Test norm binding (native F.rms_norm vs manual rsqrt fallback) and torch.compile safety.

Run: python -m pytest tests/test_norm.py -q
"""
import torch
import torch.nn.functional as F
import pytest
from nanochat import gpt


def _reference(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def test_norm_eager():
    x = torch.randn(2, 4, 16)
    y = gpt.norm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, _reference(x), atol=1e-4, rtol=1e-4)


def test_manual_fallback_branch_matches_reference():
    # Fallback path used when F.rms_norm is absent (torch<2.4)
    x = torch.randn(3, 8)
    torch.testing.assert_close(gpt._norm_manual(x), _reference(x))


@pytest.mark.skipif(not hasattr(F, "rms_norm"), reason="native F.rms_norm missing on this torch")
def test_native_branch_matches_manual():
    x = torch.randn(3, 8)
    torch.testing.assert_close(gpt._norm_native(x), gpt._norm_manual(x), atol=1e-4, rtol=1e-4)


def test_binding_matches_availability():
    expected = gpt._norm_native if hasattr(F, "rms_norm") else gpt._norm_manual
    assert gpt.norm is expected


def test_fallback_bound_when_rms_norm_absent(monkeypatch, capsys):
    # Re-import the module with F.rms_norm removed to exercise the fallback bind + echo
    import importlib
    monkeypatch.delattr(F, "rms_norm", raising=False)
    try:
        importlib.reload(gpt)
        assert gpt.norm is gpt._norm_manual
        out = capsys.readouterr().out
        assert "[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch" in out
    finally:
        monkeypatch.undo()
        importlib.reload(gpt)


def _compile_and_check(fn, backend):
    def wrapper(x):
        return fn(x) * 2

    x = torch.randn(2, 4, 16)
    try:
        out = torch.compile(wrapper, backend=backend)(x)
    except torch._dynamo.exc.InternalTorchDynamoError as e:
        pytest.fail(f"torch.compile raised InternalTorchDynamoError: {e}")
    except torch._dynamo.exc.BackendCompilerFailed as e:
        # Inductor needs a working C++ toolchain; a broken one is environmental, not a Dynamo bug
        pytest.skip(f"inductor backend unavailable on this machine: {type(e).__name__}")
    torch.testing.assert_close(out, wrapper(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fn_name", ["_norm_manual", "norm"])
def test_dynamo_traces_norm(fn_name):
    # backend="eager" exercises the Dynamo tracer only (no C++ toolchain needed)
    _compile_and_check(getattr(gpt, fn_name), backend="eager")


@pytest.mark.parametrize("fn_name", ["_norm_manual", "norm"])
def test_inductor_compiles_norm(fn_name):
    _compile_and_check(getattr(gpt, fn_name), backend="inductor")
