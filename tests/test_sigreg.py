"""
Test the SIGReg regularizer.

The statistic has a meaningful zero: it is ~0 when the samples really are N(0, I) and
grows when they are not. These tests pin that behavior down on distributions whose
answer we know in advance, which is the only cheap way to catch a sign/scale/axis bug.

python -m pytest tests/test_sigreg.py -v
"""

import pytest
import torch

from nanochat.sigreg import SIGReg, epps_pulley_statistic

N, C = 4096, 64


def stat(x, seed=0, **kw):
    sr = SIGReg(num_slices=kw.pop("num_slices", 512), **kw)
    return sr(x, generator=torch.Generator().manual_seed(seed)).item()


def gaussian(seed=0, n=N, c=C):
    return torch.randn(n, c, generator=torch.Generator().manual_seed(seed))


def test_isotropic_gaussian_is_near_zero():
    """The target distribution should sit at the floor of the statistic."""
    assert stat(gaussian()) < 5.0


def test_detects_wrong_scale():
    """N(0, 9I) is Gaussian but not *standard*: SIGReg must still reject it."""
    assert stat(3.0 * gaussian()) > 100.0


def test_detects_nonzero_mean():
    assert stat(gaussian() + 2.0) > 100.0


def test_detects_low_rank_collapse():
    """The failure mode SIGReg exists to prevent in a JEPA."""
    g = torch.Generator().manual_seed(1)
    low_rank = torch.randn(N, 2, generator=g) @ torch.randn(2, C, generator=g)
    assert stat(low_rank) > 50.0


def test_detects_anisotropy_at_matched_scale():
    """A cone-shaped cloud, rescaled to unit total variance, is still rejected."""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(N, C, generator=g)
    x[:, 0] *= 8.0  # one dominant direction
    x = x / x.std()
    assert stat(x) > stat(gaussian())


def test_scales_with_sample_count():
    """The classical statistic carries an explicit factor of n."""
    x = 3.0 * gaussian()
    small = stat(x[:1024], scale_by_n=True)
    large = stat(x, scale_by_n=True)
    assert large > 2.0 * small
    # ...and does not, when that factor is switched off
    assert stat(x[:1024], scale_by_n=False) == pytest.approx(stat(x, scale_by_n=False), rel=0.5)


def test_accepts_bt_c_and_sequence_centering():
    """(B, T, C) input, and the temporally-centered variant."""
    x = gaussian(n=32 * 64).view(32, 64, C)
    assert stat(x) < 10.0
    # a per-sequence offset is invisible to sequence centering but not to the plain test
    offset = x + torch.randn(32, 1, C, generator=torch.Generator().manual_seed(3)) * 3.0
    assert stat(offset, center="sequence") < stat(offset, center="none")


def test_gradients_flow_and_are_finite():
    x = gaussian().requires_grad_(True)
    s = stat_tensor = SIGReg(num_slices=256)(3.0 * x, generator=torch.Generator().manual_seed(0))
    s.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_gradient_descent_reaches_the_target():
    """Optimizing raw samples against SIGReg should whiten them."""
    x = (3.0 * gaussian() + 1.0).requires_grad_(True)
    sr = SIGReg(num_slices=512)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam([x], lr=0.05)
    before = sr(x, generator=torch.Generator().manual_seed(7)).item()
    for _ in range(300):
        opt.zero_grad()
        sr(x, generator=gen).backward()
        opt.step()
    after = sr(x, generator=torch.Generator().manual_seed(7)).item()
    assert after < before / 10
    assert x.mean().abs().item() < 0.15
    assert abs(x.std().item() - 1.0) < 0.15


def test_subsampling_keeps_the_estimate_in_range():
    x = 3.0 * gaussian()
    full = stat(x, scale_by_n=False)
    sub = stat(x, token_subsample=512, scale_by_n=False)
    assert sub == pytest.approx(full, rel=0.5)


def test_more_slices_lowers_variance_not_the_target():
    """num_slices controls estimator variance, not the objective itself."""
    x = 2.0 * gaussian()
    few = [stat(x, num_slices=8, seed=s) for s in range(8)]
    many = [stat(x, num_slices=1024, seed=s) for s in range(8)]
    spread = lambda v: max(v) - min(v)
    assert spread(many) < spread(few)
    assert sum(many) / len(many) == pytest.approx(sum(few) / len(few), rel=0.3)


def test_epps_pulley_statistic_shape():
    sr = SIGReg(num_slices=16, knots=17)
    z = torch.randn(256, 16)
    out = epps_pulley_statistic(z, sr.t, sr.phi, sr.weights)
    assert out.shape == (16,)
    assert (out >= 0).all()  # it is an integral of a squared modulus
