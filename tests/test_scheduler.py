
import pytest
from nanochat.scheduler import get_lr_multiplier, get_muon_momentum

def test_adamw_warmup_linear():
    """Test that learning rate warms up linearly."""
    num_iterations = 100
    warmup_ratio = 0.1 # 10 steps
    warmdown_ratio = 0.0 # disable warmdown for this test
    final_lr_frac = 0.0

    # Step 0: should be 1/10 = 0.1
    lr_0 = get_lr_multiplier(0, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_0 == pytest.approx(0.1)

    # Step 4: should be 5/10 = 0.5
    lr_4 = get_lr_multiplier(4, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_4 == pytest.approx(0.5)

    # Step 9: should be 10/10 = 1.0
    lr_9 = get_lr_multiplier(9, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_9 == pytest.approx(1.0)

    # Step 10: should be 1.0 (past warmup)
    lr_10 = get_lr_multiplier(10, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_10 == pytest.approx(1.0)

def test_no_warmup():
    """Test that 0 warmup ratio gives 1.0 immediately."""
    num_iterations = 100
    warmup_ratio = 0.0
    warmdown_ratio = 0.0
    final_lr_frac = 0.0

    lr_0 = get_lr_multiplier(0, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_0 == pytest.approx(1.0)

def test_warmdown():
    """Test that learning rate decays during warmdown."""
    num_iterations = 100
    warmup_ratio = 0.0
    warmdown_ratio = 0.1 # 10 steps at the end
    final_lr_frac = 0.1

    # Start of warmdown: num_iterations - warmdown_iters = 90
    # Step 90: start of decay.
    # it <= 90 (100 - 10). If it is exactly 90, it returns 1.0 because of <= check in code?
    # Logic: elif it <= num_iterations - warmdown_iters: return 1.0
    # So step 90 is 1.0.
    lr_90 = get_lr_multiplier(90, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_90 == pytest.approx(1.0)

    # Step 91. progress = (100 - 91) / 10 = 0.9.
    # Result = 0.9 * 1.0 + 0.1 * 0.1 = 0.9 + 0.01 = 0.91
    lr_91 = get_lr_multiplier(91, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_91 == pytest.approx(0.91)

    # Step 99. progress = (100 - 99) / 10 = 0.1.
    # Result = 0.1 * 1.0 + 0.9 * 0.1 = 0.1 + 0.09 = 0.19
    lr_99 = get_lr_multiplier(99, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    assert lr_99 == pytest.approx(0.19)

def test_muon_momentum():
    """Test Muon momentum ramp-up."""
    # frac = min(it / 300, 1)
    # momentum = (1 - frac) * 0.85 + frac * 0.95

    # Step 0: frac=0 -> 0.85
    m_0 = get_muon_momentum(0)
    assert m_0 == pytest.approx(0.85)

    # Step 150: frac=0.5 -> 0.90
    m_150 = get_muon_momentum(150)
    assert m_150 == pytest.approx(0.90)

    # Step 300: frac=1.0 -> 0.95
    m_300 = get_muon_momentum(300)
    assert m_300 == pytest.approx(0.95)

    # Step 400: frac=1.0 -> 0.95
    m_400 = get_muon_momentum(400)
    assert m_400 == pytest.approx(0.95)
