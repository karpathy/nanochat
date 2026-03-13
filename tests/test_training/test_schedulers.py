"""Test learning rate schedulers."""
import math


def test_lr_warmup():
    """Test LR warmup schedule."""
    def get_lr_multiplier(it, warmup_steps=40, num_iterations=1000, warmdown_ratio=0.65, final_lr_frac=0.05):
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_steps:
            return (it + 1) / warmup_steps
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac
    
    # Test warmup phase
    assert get_lr_multiplier(0) < 1.0
    assert get_lr_multiplier(20) < 1.0
    assert abs(get_lr_multiplier(39) - 1.0) < 0.05
    
    # Test constant phase
    assert get_lr_multiplier(100) == 1.0
    assert get_lr_multiplier(300) == 1.0
    
    # Test warmdown phase
    assert get_lr_multiplier(900) < 1.0
    assert get_lr_multiplier(999) < 0.1


def test_muon_momentum_schedule():
    """Test Muon momentum schedule."""
    def get_muon_momentum(it):
        frac = min(it / 400, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.97
        return momentum
    
    # Test initial momentum
    assert abs(get_muon_momentum(0) - 0.85) < 0.01
    
    # Test ramp up
    assert 0.85 < get_muon_momentum(200) < 0.97
    
    # Test final momentum
    assert abs(get_muon_momentum(400) - 0.97) < 0.01
    assert abs(get_muon_momentum(1000) - 0.97) < 0.01


def test_weight_decay_schedule():
    """Test weight decay cosine schedule."""
    def get_weight_decay(it, weight_decay_scaled=0.28, num_iterations=1000):
        return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))
    
    # Test initial weight decay
    assert abs(get_weight_decay(0, 0.28, 1000) - 0.28) < 0.01
    
    # Test middle
    mid_wd = get_weight_decay(500, 0.28, 1000)
    assert 0.0 < mid_wd < 0.28
    
    # Test final (should be near zero)
    assert get_weight_decay(999, 0.28, 1000) < 0.01
