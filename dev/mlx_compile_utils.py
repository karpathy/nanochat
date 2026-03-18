from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def build_loss_and_grad(model) -> Callable:
    return nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))


def make_training_step(model, optimizer, *, execution_mode: str) -> Callable:
    if execution_mode not in {"eager", "compiled"}:
        raise ValueError(f"unsupported execution mode: {execution_mode}")

    loss_and_grad = build_loss_and_grad(model)
    if execution_mode == "eager":
        return loss_and_grad

    captured_state = {
        "model": model.state,
        "optimizer": optimizer.state_trees(),
    }

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
    mx.eval(loss, *extra_tensors, model.parameters(), *optimizer.state_trees())