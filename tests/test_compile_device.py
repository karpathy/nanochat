"""Inductor (default torch.compile backend) has no MPS codegen — we must stay eager there."""

import torch
import torch.nn as nn

from nanochat.common import inductor_supports_device, maybe_torch_compile
from nanochat import optim as optim_mod


def test_inductor_supported_on_cuda_and_cpu_not_mps():
    assert inductor_supports_device("cuda")
    assert inductor_supports_device("cpu")
    assert not inductor_supports_device("mps")


def test_maybe_torch_compile_leaves_module_eager_on_mps():
    module = nn.Linear(4, 4)
    out = maybe_torch_compile(module, "mps", dynamic=False)
    assert out is module


def test_maybe_torch_compile_wraps_on_cpu(monkeypatch):
    module = nn.Linear(4, 4)
    sentinel = object()

    def fake_compile(m, **kwargs):
        assert m is module
        assert kwargs.get("dynamic") is False
        return sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert maybe_torch_compile(module, "cpu", dynamic=False) is sentinel


def test_adamw_step_fused_uses_eager_impl_on_mps(monkeypatch):
    calls = {"eager": 0, "compiled": 0}

    def fake_eager(*args, **kwargs):
        calls["eager"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_adamw_step_impl", fake_eager)
    monkeypatch.setattr(optim_mod, "_adamw_step_compiled", fake_compiled)

    class _Dev:
        type = "mps"

    class _T:
        device = _Dev()

    optim_mod.adamw_step_fused(_T(), None, None, None, None, None, None, None, None, None)
    assert calls == {"eager": 1, "compiled": 0}


def test_adamw_step_fused_uses_compiled_impl_off_mps(monkeypatch):
    calls = {"eager": 0, "compiled": 0}

    def fake_eager(*args, **kwargs):
        calls["eager"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_adamw_step_impl", fake_eager)
    monkeypatch.setattr(optim_mod, "_adamw_step_compiled", fake_compiled)

    class _Dev:
        type = "cpu"

    class _T:
        device = _Dev()

    optim_mod.adamw_step_fused(_T(), None, None, None, None, None, None, None, None, None)
    assert calls == {"eager": 0, "compiled": 1}
