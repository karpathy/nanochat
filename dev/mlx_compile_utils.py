from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def build_loss_and_grad(model) -> Callable:
    return nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))


def capture_optimizer_state(optimizer):
    state_trees = getattr(optimizer, "state_trees", None)
    if callable(state_trees):
        return state_trees()
    return optimizer.state


def iter_optimizer_state(optimizer) -> tuple[object, ...]:
    state = capture_optimizer_state(optimizer)
    if isinstance(state, tuple):
        return state
    if isinstance(state, list):
        return tuple(state)
    return (state,)


def capture_training_state(model, optimizer, *, include_random_state: bool = False) -> dict[str, object]:
    """Return the mutable state that compiled MLX training steps must capture explicitly."""
    captured_state: dict[str, object] = {
        "model": model.state,
        "optimizer": capture_optimizer_state(optimizer),
    }
    if include_random_state:
        # Required if future compiled hot paths add dropout or other stochastic MLX ops.
        captured_state["random"] = mx.random.state
    return captured_state


def make_training_step(model, optimizer, *, execution_mode: str) -> Callable:
    if execution_mode not in {"eager", "compiled"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")

    loss_and_grad = build_loss_and_grad(model)
    if execution_mode == "eager":
        return loss_and_grad

    captured_state = capture_training_state(model, optimizer)

    def compiled_step(batch, labels):
        loss, grads = loss_and_grad(batch, labels)
        optimizer.update(model, grads)
        return loss, grads

    return mx.compile(
        compiled_step,
        inputs=captured_state,
        outputs=captured_state,
    )


def eval_training_state(loss, model, optimizer, *extra_tensors) -> None:
    mx.eval(loss, *extra_tensors, model.parameters(), *iter_optimizer_state(optimizer))