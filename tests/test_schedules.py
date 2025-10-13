import torch
import pytest

from nanochat.schedules import compute_lr_multiplier, apply_lr_multiplier

def test_compute_lr_multiplier_handles_warmup():
    multiplier = compute_lr_multiplier(0, 100, warmup_ratio=0.1)
    assert multiplier == pytest.approx(0.1)

def test_compute_lr_multiplier_handles_warmdown():
    multiplier = compute_lr_multiplier(95, 100, warmdown_ratio=0.1, final_lr_frac=0.1)
    # progress = (100-95)/10 = 0.5 -> 0.5 + 0.5*0.1
    assert multiplier == pytest.approx(0.55)

def test_apply_lr_multiplier_uses_initial_lr():
    param = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.SGD([param], lr=0.2)
    apply_lr_multiplier(opt, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    assert opt.param_groups[0]["initial_lr"] == pytest.approx(0.2)
    apply_lr_multiplier(opt, 1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.2)
