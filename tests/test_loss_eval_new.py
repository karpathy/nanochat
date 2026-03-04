import math

import torch

import nanochat.loss_eval as loss_eval


class FakeModel:
    def get_device(self):
        return torch.device("cpu")

    def __call__(self, x, y, loss_reduction="none"):
        del x, loss_reduction
        # Return a deterministic per-token loss tensor matching y shape.
        return torch.ones_like(y, dtype=torch.float32)


def test_evaluate_bpb_paths(monkeypatch):
    model = FakeModel()
    token_bytes = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

    # Path with ignored targets (<0).
    batches1 = [
        (
            torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            torch.tensor([[1, -1], [2, 0]], dtype=torch.long),
        )
    ]
    monkeypatch.setattr(loss_eval.dist, "is_initialized", lambda: False)
    out1 = loss_eval.evaluate_bpb(model, batches1, steps=1, token_bytes=token_bytes)
    assert out1 > 0.0

    # Fast path without ignored targets.
    batches2 = [
        (
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[3, 4]], dtype=torch.long),
        )
    ]
    out2 = loss_eval.evaluate_bpb(model, batches2, steps=1, token_bytes=token_bytes)
    assert out2 > 0.0

    # Distributed reduction path.
    calls = {"n": 0}
    monkeypatch.setattr(loss_eval.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loss_eval.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(loss_eval.dist, "all_reduce", lambda *_a, **_k: calls.__setitem__("n", calls["n"] + 1))
    out3 = loss_eval.evaluate_bpb(model, batches2, steps=1, token_bytes=token_bytes)
    assert out3 > 0.0
    assert calls["n"] == 2

    # No counted bytes -> inf.
    zero_bytes = torch.zeros_like(token_bytes)
    out4 = loss_eval.evaluate_bpb(model, batches2, steps=1, token_bytes=zero_bytes)
    assert out4 == float("inf")

