import pytest

pytest.importorskip("mlx.core")

from dev import mlx_compile_utils


class FakeModel:
    def __init__(self):
        self.state = object()


class FakeOptimizer:
    def __init__(self):
        self._state = object()
        self.updates = []

    def state_trees(self):
        return self._state

    def update(self, model, grads):
        self.updates.append((model, grads))


def test_make_training_step_compiled_captures_explicit_state(monkeypatch):
    model = FakeModel()
    optimizer = FakeOptimizer()
    compile_call = {}

    def fake_loss_and_grad(batch, labels):
        return "loss", "grads"

    def fake_compile(fn, *, inputs, outputs):
        compile_call["inputs"] = inputs
        compile_call["outputs"] = outputs

        def wrapped(*args):
            return fn(*args)

        return wrapped

    monkeypatch.setattr(mlx_compile_utils, "build_loss_and_grad", lambda _: fake_loss_and_grad)
    monkeypatch.setattr(mlx_compile_utils.mx, "compile", fake_compile)

    train_step = mlx_compile_utils.make_training_step(model, optimizer, execution_mode="compiled")
    loss, grads = train_step("batch", "labels")

    assert loss == "loss"
    assert grads == "grads"
    assert compile_call["inputs"] is compile_call["outputs"]
    assert compile_call["inputs"] == {
        "model": model.state,
        "optimizer": optimizer.state_trees(),
    }
    assert optimizer.updates == [(model, "grads")]