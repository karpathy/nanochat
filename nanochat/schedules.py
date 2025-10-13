"""Learning rate schedule utilities."""


def compute_lr_multiplier(
    step: int,
    total_steps: int,
    *,
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.0,
    final_lr_frac: float = 0.0,
) -> float:
    """Compute LR multiplier with linear warmup and warmdown phases.

    The multiplier ramps linearly from 0 -> 1 during warmup, stays at 1, then
    decays linearly to ``final_lr_frac`` during warmdown. Ratios are expressed
    as fractions of ``total_steps``.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    step = min(step, total_steps)
    warmup_steps = int(round(warmup_ratio * total_steps))
    warmdown_steps = int(round(warmdown_ratio * total_steps))

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps

    if warmdown_steps > 0 and step >= total_steps - warmdown_steps:
        progress = (total_steps - step) / max(1, warmdown_steps)
        return progress + (1 - progress) * final_lr_frac

    return 1.0


def apply_lr_multiplier(
    optimizer,
    multiplier: float,
    *,
    base_key: str = "initial_lr",
) -> float:
    """Apply ``multiplier`` to an optimizer in-place using ``base_key`` as base LR."""

    for group in optimizer.param_groups:
        base_lr = group.get(base_key)
        if base_lr is None:
            base_lr = group["lr"]
            group[base_key] = base_lr
        group["lr"] = base_lr * multiplier
    return multiplier
