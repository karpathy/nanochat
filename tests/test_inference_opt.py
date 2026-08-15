"""
Tests for inference-path optimizations (last-token logits, fused QKV, RoPE).
"""

import torch

from nanochat.engine import KVCache, sample_next_token
from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb


def _tiny_model():
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    model.eval()
    return model


@torch.no_grad()
def test_apply_rotary_emb_matches_cat_reference():
    B, T, H, D = 2, 5, 3, 8
    x = torch.randn(B, T, H, D)
    cos = torch.randn(1, T, 1, D // 2)
    sin = torch.randn(1, T, 1, D // 2)
    d = D // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    expected = torch.cat([y1, y2], 3).to(x.dtype)
    actual = apply_rotary_emb(x, cos, sin)
    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_last_logits_only_matches_full_forward():
    model = _tiny_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 7))
    full = model.forward(idx)
    last = model.forward(idx, last_logits_only=True)
    assert last.shape == (2, 1, model.config.vocab_size)
    torch.testing.assert_close(last, full[:, -1:, :])


@torch.no_grad()
def test_kv_cache_prefill_returns_last_token_logits():
    model = _tiny_model()
    B, T = 2, 6
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    full = model.forward(idx)
    kv = KVCache(
        batch_size=B,
        num_heads=model.config.n_kv_head,
        seq_len=T,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
    )
    cached = model.forward(idx, kv_cache=kv)
    assert cached.shape == (B, 1, model.config.vocab_size)
    torch.testing.assert_close(cached, full[:, -1:, :], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_fuse_qkv_matches_unfused_logits():
    model = _tiny_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 5))
    before = model.forward(idx)
    model.prepare_for_inference()
    after = model.forward(idx)
    torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_sample_next_token_accepts_bf16_logits():
    logits = torch.randn(3, 32, dtype=torch.bfloat16)
    rng = torch.Generator()
    rng.manual_seed(0)
    greedy = sample_next_token(logits, rng, temperature=0.0)
    assert greedy.shape == (3, 1)
    sampled = sample_next_token(logits, rng, temperature=0.8, top_k=8)
    assert sampled.shape == (3, 1)
    assert sampled.min() >= 0 and sampled.max() < 32
