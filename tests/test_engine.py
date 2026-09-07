"""
The inference engine, on a tiny real GPT in fp32 on CPU: decoding token by token
through the KV cache must give the logits of a full forward pass, and sampling must
be seeded, independent per row, and greedy at temperature 0.

python -m pytest tests/test_engine.py -v
"""

import pytest
import torch

import nanochat.gpt
import nanochat.engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache, Engine

VOCAB = 64
ASSISTANT_END = 62
BOS = 63


class Specials:
    """The two special tokens the engine asks the tokenizer for."""
    def encode_special(self, s):
        assert s == "<|assistant_end|>"
        return ASSISTANT_END
    def get_bos_token_id(self):
        return BOS


@pytest.fixture
def model(monkeypatch):
    """A 2-layer GPT with GQA and one sliding-window layer, in fp32 on CPU so the two
    code paths can be compared tightly (the compute dtype is a module constant)."""
    monkeypatch.setattr(nanochat.gpt, "COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr(nanochat.engine, "COMPUTE_DTYPE", torch.float32)
    config = GPTConfig(sequence_len=256, vocab_size=VOCAB, n_layer=2, n_head=2, n_kv_head=1, n_embd=32,
                       window_pattern="SL", short_seq_len=128) # the short window must be a multiple of 128
    torch.manual_seed(0)
    return GPT(config, device="cpu")


def test_kv_cache_decode_matches_full_forward(model):
    T = 160 # past the short window, so the windowed layer is exercised during decode
    ids = torch.randint(0, VOCAB, (1, T), generator=torch.Generator().manual_seed(1))
    full = model(ids) # (1, T, V), every position at once
    m = model.config
    cache = KVCache(batch_size=1, num_heads=m.n_kv_head, seq_len=T, head_dim=m.n_embd // m.n_head,
                    num_layers=m.n_layer, device="cpu", dtype=torch.float32)
    T_prefill = 130
    prefill = model(ids[:, :T_prefill], kv_cache=cache)
    torch.testing.assert_close(prefill, full[:, :T_prefill], atol=1e-4, rtol=1e-4)
    for t in range(T_prefill, T):
        step = model(ids[:, t:t + 1], kv_cache=cache) # (1, 1, V)
        torch.testing.assert_close(step[:, 0], full[:, t], atol=1e-4, rtol=1e-4)
    assert cache.get_pos() == T


def test_generate_is_seeded_and_greedy_at_zero_temperature(model):
    engine = Engine(model, Specials())
    prompt = [BOS, 5, 17, 40, 3]
    # temperature 0: every sample is the greedy continuation, and the first token is the
    # argmax of the model's own logits (this also checks the prefill cache is replicated per row)
    rows = engine.generate_batch(prompt, num_samples=3, max_tokens=6, temperature=0.0)
    greedy_first = model(torch.tensor([prompt]))[0, -1].argmax().item()
    assert all(row[:len(prompt)] == prompt for row in rows)
    assert all(row == rows[0] for row in rows)
    assert rows[0][len(prompt)] == greedy_first
    assert all(len(row) - len(prompt) <= 6 for row in rows)
    # temperature 1: seeded, and the rows are sampled independently, not broadcast
    a = engine.generate_batch(prompt, num_samples=8, max_tokens=6, temperature=1.0, seed=1)
    b = engine.generate_batch(prompt, num_samples=8, max_tokens=6, temperature=1.0, seed=1)
    c = engine.generate_batch(prompt, num_samples=8, max_tokens=6, temperature=1.0, seed=2)
    assert a == b
    assert a != c
    first_tokens = {row[len(prompt)] for row in a if len(row) > len(prompt)}
    assert len(first_tokens) > 1, "all 8 rows sampled the same first token: broadcast instead of sampled"
