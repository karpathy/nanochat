
import math

def get_lr_multiplier(it, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac):
    """
    Calculates the learning rate multiplier based on the current iteration.

    Args:
        it (int): Current iteration step.
        warmup_ratio (float): Ratio of total iterations for warmup.
        num_iterations (int): Total number of training iterations.
        warmdown_ratio (float): Ratio of total iterations for warmdown.
        final_lr_frac (float): Fraction of initial learning rate at the end of training.

    Returns:
        float: Learning rate multiplier.
    """
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it, ramp_steps=300):
    """
    Calculates the momentum for Muon optimizer.

    Args:
        it (int): Current iteration step.
        ramp_steps (int): Number of steps to ramp up momentum.

    Returns:
        float: Momentum value.
    """
    frac = min(it / ramp_steps, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
