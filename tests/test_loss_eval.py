from types import SimpleNamespace

import pytest
import torch

from nanochat.loss_eval import assert_causal_logits


class ToyModel:
    def __init__(self, leak=None, chunk_size=None):
        self.config = SimpleNamespace(sequence_len=64, vocab_size=64)
        self.leak = leak
        self.chunk_size = chunk_size

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, tokens):
        logits = tokens.float().unsqueeze(-1)
        if self.leak == "global":
            logits = logits + tokens.float().mean(dim=1, keepdim=True).unsqueeze(-1)
        elif self.leak == "chunk":
            batch_size, sequence_len = tokens.shape
            chunks = tokens.float().view(
                batch_size, sequence_len // self.chunk_size, self.chunk_size
            )
            summaries = chunks.mean(dim=2, keepdim=True).expand_as(chunks)
            logits = logits + summaries.reshape(batch_size, sequence_len).unsqueeze(-1)
        elif self.leak == "circular":
            logits = logits + torch.roll(tokens.float(), shifts=-1, dims=1).unsqueeze(-1)
        return logits


def test_causal_logits_reject_suffix_leakage():
    assert_causal_logits(ToyModel())
    with pytest.raises(RuntimeError, match="suffix tokens changed prefix logits"):
        assert_causal_logits(ToyModel(leak="global"))


@pytest.mark.parametrize("chunk_size", [8, 16, 32, 64])
def test_causal_logits_reject_aligned_chunk_leakage(chunk_size):
    with pytest.raises(RuntimeError, match="suffix tokens changed prefix logits"):
        assert_causal_logits(ToyModel(leak="chunk", chunk_size=chunk_size))


def test_causal_logits_reject_circular_leakage():
    with pytest.raises(RuntimeError, match="suffix tokens changed prefix logits"):
        assert_causal_logits(ToyModel(leak="circular"))
