from types import SimpleNamespace

import pytest
import torch

from nanochat.loss_eval import assert_causal_logits


class ToyModel:
    def __init__(self, leak=False):
        self.config = SimpleNamespace(sequence_len=64, vocab_size=64)
        self.leak = leak

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, tokens):
        logits = tokens.float().unsqueeze(-1)
        if self.leak:
            logits = logits + tokens.float().mean(dim=1, keepdim=True).unsqueeze(-1)
        return logits


def test_causal_logits_reject_suffix_leakage():
    assert_causal_logits(ToyModel())
    with pytest.raises(RuntimeError, match="suffix tokens changed prefix logits"):
        assert_causal_logits(ToyModel(leak=True))
