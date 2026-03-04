from types import SimpleNamespace

import pytest
import torch

import nanochat.optim as optim


def test_fused_step_functions_via_wrapped():
    # AdamW fused kernel.
    p = torch.ones(2, 2, dtype=torch.float32)
    g = torch.full((2, 2), 0.1, dtype=torch.float32)
    exp_avg = torch.zeros_like(p)
    exp_avg_sq = torch.zeros_like(p)
    optim.adamw_step_fused.__wrapped__(
        p,
        g,
        exp_avg,
        exp_avg_sq,
        torch.tensor(1.0),
        torch.tensor(0.01),
        torch.tensor(0.9),
        torch.tensor(0.99),
        torch.tensor(1e-8),
        torch.tensor(0.01),
    )
    assert not torch.equal(p, torch.ones_like(p))

    # Muon fused kernel: tall matrix branch.
    grads_tall = torch.randn(2, 4, 2, dtype=torch.float32)
    params_tall = torch.randn(2, 4, 2, dtype=torch.float32)
    m_tall = torch.zeros_like(grads_tall)
    v_tall = torch.zeros(2, 4, 1, dtype=torch.float32)
    optim.muon_step_fused.__wrapped__(
        grads_tall,
        params_tall,
        m_tall,
        v_tall,
        torch.tensor(0.9),
        torch.tensor(0.01),
        torch.tensor(0.0),
        torch.tensor(0.95),
        2,
        -1,
    )

    # Muon fused kernel: wide matrix branch.
    grads_wide = torch.randn(2, 2, 4, dtype=torch.float32)
    params_wide = torch.randn(2, 2, 4, dtype=torch.float32)
    m_wide = torch.zeros_like(grads_wide)
    v_wide = torch.zeros(2, 1, 4, dtype=torch.float32)
    optim.muon_step_fused.__wrapped__(
        grads_wide,
        params_wide,
        m_wide,
        v_wide,
        torch.tensor(0.9),
        torch.tensor(0.01),
        torch.tensor(0.0),
        torch.tensor(0.95),
        2,
        -2,
    )


def test_muon_adamw_optimizer_paths(monkeypatch):
    monkeypatch.setattr(
        optim,
        "adamw_step_fused",
        lambda p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t: p.data.add_(-grad * lr_t.item()),
    )
    monkeypatch.setattr(
        optim,
        "muon_step_fused",
        lambda stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim: stacked_params.add_(-stacked_grads * lr_t.item()),
    )

    p1 = torch.nn.Parameter(torch.ones(4, 4))
    p2 = torch.nn.Parameter(torch.ones(4, 4))
    p1.grad = torch.full_like(p1, 0.1)
    p2.grad = torch.full_like(p2, 0.2)

    p1_nograd = torch.nn.Parameter(torch.ones(4, 4))
    groups = [
        dict(kind="adamw", params=[p1_nograd, p1], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(kind="muon", params=[p2], lr=0.02, momentum=0.9, ns_steps=2, beta2=None, weight_decay=0.0),
    ]
    opt = optim.MuonAdamW(groups)
    opt.step()
    assert p1.data.mean().item() < 1.0
    assert p2.data.mean().item() < 1.0

    # _step_muon early return on empty params.
    opt._step_muon(dict(kind="muon", params=[], lr=0.01, momentum=0.9, ns_steps=1, beta2=0.95, weight_decay=0.0))

    bad = optim.MuonAdamW([dict(kind="bad", params=[p1], lr=0.1)])
    with pytest.raises(ValueError):
        bad.step()


class _Future:
    def __init__(self):
        self.waited = 0

    def wait(self):
        self.waited += 1
        return None


class _AsyncOp:
    def __init__(self):
        self.f = _Future()

    def get_future(self):
        return self.f


def test_dist_muon_adamw_components(monkeypatch):
    # Fake distributed ops.
    def fake_all_reduce(t, op=None, async_op=False):
        del op, async_op
        return _AsyncOp()

    def fake_reduce_scatter_tensor(out, inp, op=None, async_op=False):
        del op, async_op
        # take the leading chunk
        flat = inp.reshape(inp.shape[0], -1)
        out.copy_(inp[: out.shape[0]])
        return _AsyncOp()

    def fake_all_gather_into_tensor(out, inp, async_op=False):
        del async_op
        # repeat input chunks into output prefix as needed
        n = inp.shape[0]
        out.data[:n].copy_(inp.detach())
        if out.shape[0] > n:
            out.data[n:].zero_()
        return _AsyncOp()

    monkeypatch.setattr(optim.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(optim.dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    monkeypatch.setattr(optim.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(optim.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(optim.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        optim,
        "adamw_step_fused",
        lambda p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t: p.data.add_(-grad * lr_t.item()),
    )
    monkeypatch.setattr(
        optim,
        "muon_step_fused",
        lambda stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim: stacked_params.add_(-stacked_grads * lr_t.item()),
    )

    p_small = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    p_small.grad = torch.full_like(p_small, 0.1)
    p_large = torch.nn.Parameter(torch.ones(1024, 2), requires_grad=False)
    p_large.grad = torch.full_like(p_large, 0.1)

    p_mu0 = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p_mu1 = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p_mu2 = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    for p in (p_mu0, p_mu1, p_mu2):
        p.grad = torch.full_like(p, 0.2)

    groups = [
        dict(kind="adamw", params=[p_small, p_large], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(kind="muon", params=[p_mu0, p_mu1, p_mu2], lr=0.02, momentum=0.9, ns_steps=2, beta2=0.95, weight_decay=0.0),
    ]
    opt = optim.DistMuonAdamW(groups)

    # _reduce_adamw small and large paths.
    info_adam = opt._reduce_adamw(groups[0], world_size=2)
    assert len(info_adam["param_infos"]) == 2
    assert info_adam["param_infos"][p_small]["is_small"] is True
    assert info_adam["param_infos"][p_large]["is_small"] is False

    # Assertion on non-divisible shape[0] for large params.
    bad_p = torch.nn.Parameter(torch.ones(1025, 2), requires_grad=False)
    bad_p.grad = torch.ones_like(bad_p)
    with pytest.raises(AssertionError):
        opt._reduce_adamw(dict(kind="adamw", params=[bad_p], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0), world_size=2)

    info_mu = opt._reduce_muon(groups[1], world_size=2)
    assert "chunk_size" in info_mu

    gather_list = []
    opt._compute_adamw(groups[0], info_adam, gather_list, rank=1, world_size=2)
    assert len(gather_list) >= 1

    opt._compute_muon(groups[1], info_mu, gather_list, rank=1)
    assert len(gather_list) >= 2
    opt._compute_muon(groups[1], info_mu, gather_list, rank=3)

    # _finish_gathers handles both params=None and params=list branches.
    gather_list.append(dict(future=_Future(), params=None))
    opt._finish_gathers(gather_list)

    # step() happy path.
    opt.step()

    # step() unknown kind.
    bad = optim.DistMuonAdamW([dict(kind="bad", params=[p_small], lr=0.1)])
    with pytest.raises(ValueError):
        bad.step()


def test_dist_optimizer_phase2_unknown_kind(monkeypatch):
    p = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    p.grad = torch.ones_like(p)
    group = dict(kind="muon", params=[p], lr=0.01, momentum=0.9, ns_steps=1, beta2=0.95, weight_decay=0.0)
    opt = optim.DistMuonAdamW([group])

    monkeypatch.setattr(optim.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim.dist, "get_world_size", lambda: 1)

    def mutate_kind(g, world_size):
        del world_size
        g["kind"] = "bad"
        return {}

    monkeypatch.setattr(opt, "_reduce_muon", mutate_kind)
    monkeypatch.setattr(opt, "_compute_adamw", lambda *a, **k: None)
    monkeypatch.setattr(opt, "_compute_muon", lambda *a, **k: None)
    monkeypatch.setattr(opt, "_finish_gathers", lambda *a, **k: None)

    with pytest.raises(ValueError):
        opt.step()
