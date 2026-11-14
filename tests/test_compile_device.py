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


def test_adamw_step_fused_uses_mps_impl_on_mps(monkeypatch):
    calls = {"mps": 0, "compiled": 0}

    def fake_mps(*args, **kwargs):
        calls["mps"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_adamw_step_mps", fake_mps)
    monkeypatch.setattr(optim_mod, "_adamw_step_compiled", fake_compiled)

    class _Dev:
        type = "mps"

    class _T:
        device = _Dev()

    optim_mod.adamw_step_fused(_T(), None, None, None, None, None, None, None, None, None)
    assert calls == {"mps": 1, "compiled": 0}


def test_adamw_step_fused_uses_compiled_impl_off_mps(monkeypatch):
    calls = {"mps": 0, "compiled": 0}

    def fake_mps(*args, **kwargs):
        calls["mps"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_adamw_step_mps", fake_mps)
    monkeypatch.setattr(optim_mod, "_adamw_step_compiled", fake_compiled)

    class _Dev:
        type = "cpu"

    class _T:
        device = _Dev()

    optim_mod.adamw_step_fused(_T(), None, None, None, None, None, None, None, None, None)
    assert calls == {"mps": 0, "compiled": 1}


def test_adamw_step_mps_accepts_cpu_scalar_hyperparams():
    """MPS rejects 0-D CPU tensors in lerp_; the MPS path must use Python floats."""
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    device = torch.device("mps")
    p = torch.randn(4, 4, device=device)
    grad = torch.randn(4, 4, device=device)
    exp_avg = torch.zeros_like(p)
    exp_avg_sq = torch.zeros_like(p)
    before = p.clone()
    optim_mod._adamw_step_mps(
        p, grad, exp_avg, exp_avg_sq,
        torch.tensor(1.0), torch.tensor(1e-3), torch.tensor(0.9),
        torch.tensor(0.95), torch.tensor(1e-8), torch.tensor(0.0),
    )
    assert not torch.equal(p, before)


def test_muon_step_fused_uses_mps_impl_on_mps(monkeypatch):
    calls = {"mps": 0, "compiled": 0}

    def fake_mps(*args, **kwargs):
        calls["mps"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_muon_step_mps", fake_mps)
    monkeypatch.setattr(optim_mod, "_muon_step_compiled", fake_compiled)

    class _Dev:
        type = "mps"

    class _T:
        device = _Dev()

    optim_mod.muon_step_fused(_T(), _T(), None, None, None, None, None, None, 5, -1)
    assert calls == {"mps": 1, "compiled": 0}


def test_muon_step_fused_uses_compiled_impl_off_mps(monkeypatch):
    calls = {"mps": 0, "compiled": 0}

    def fake_mps(*args, **kwargs):
        calls["mps"] += 1

    def fake_compiled(*args, **kwargs):
        calls["compiled"] += 1

    monkeypatch.setattr(optim_mod, "_muon_step_mps", fake_mps)
    monkeypatch.setattr(optim_mod, "_muon_step_compiled", fake_compiled)

    class _Dev:
        type = "cpu"

    class _T:
        device = _Dev()

    optim_mod.muon_step_fused(_T(), _T(), None, None, None, None, None, None, 5, -1)
    assert calls == {"mps": 0, "compiled": 1}


def test_muon_step_mps_accepts_cpu_scalar_hyperparams():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    device = torch.device("mps")
    # Wide matrices exercise the wide Polar Express branch (size(-2) <= size(-1)).
    stacked_grads = torch.randn(2, 4, 8, device=device)
    stacked_params = torch.randn(2, 4, 8, device=device)
    momentum_buffer = torch.zeros_like(stacked_grads)
    second_momentum_buffer = torch.zeros(2, 1, 8, device=device)
    before = stacked_params.clone()
    optim_mod._muon_step_mps(
        stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
        torch.tensor(0.95), torch.tensor(0.02), torch.tensor(0.0), torch.tensor(0.9),
        ns_steps=2, red_dim=-2,
    )
    assert torch.isfinite(stacked_params).all()
    assert not torch.equal(stacked_params, before)


def test_finish_gathers_copies_one_by_one_on_mps():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    device = torch.device("mps")
    p0 = torch.zeros(2, 3, device=device)
    p1 = torch.zeros(2, 3, device=device)
    stacked = torch.stack([torch.ones(2, 3, device=device), torch.full((2, 3), 2.0, device=device)])
    opt = optim_mod.MuonAdamW([{"params": [p0, p1], "kind": "muon", "lr": 0.02, "momentum": 0.95, "ns_steps": 1, "beta2": 0.9, "weight_decay": 0.0}])
    opt._finish_gathers([{"future": None, "params": [p0, p1], "stacked_params": stacked}])
    assert torch.equal(p0, stacked[0]) and torch.equal(p1, stacked[1])


def test_finish_gathers_skips_empty_params():
    opt = optim_mod.MuonAdamW([{"params": [torch.zeros(1)], "kind": "adamw", "lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0}])
    opt._finish_gathers([{"future": None, "params": [], "stacked_params": torch.zeros(0)}])
