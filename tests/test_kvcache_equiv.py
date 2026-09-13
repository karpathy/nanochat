"""Checks that decoding with a KV cache gives the same logits as a full forward pass.

Both paths run on a small real GPT, no tokenizer or dataset needed:
  1) one full forward over the whole sequence
  2) prefill the prompt into a cache, then feed one token at a time
The last-position logits must match (atol=1e-4). This also covers RoPE position
slicing: if the cache path used the wrong absolute positions, they would diverge.
"""
import torch

from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


def _tiny_model():
    cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    torch.manual_seed(0)
    model = GPT(cfg)
    model.eval()
    return model, cfg


def _kv_kwargs(cfg):
    return {"num_heads": cfg.n_kv_head, "head_dim": cfg.n_embd // cfg.n_head,
            "num_layers": cfg.n_layer}


@torch.no_grad()
def test_prefill_matches_full_forward():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))

    full = model(tokens)
    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))
    prefill = model(tokens, kv_cache=cache)

    assert torch.allclose(full[:, -1], prefill[:, -1], atol=1e-4)


@torch.no_grad()
def test_decode_one_token_matches_full_forward():
    model, cfg = _tiny_model()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))

    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))
    model(tokens, kv_cache=cache)        # prefill
    decode = model(nxt, kv_cache=cache)  # one step of decode

    full = model(torch.cat([tokens, nxt], dim=1))
    assert torch.allclose(decode[:, -1], full[:, -1], atol=1e-4)


@torch.no_grad()
def test_cache_position_advances():
    model, cfg = _tiny_model()
    cache = KVCache(batch_size=1, seq_len=32, device="cpu", dtype=torch.float32, **_kv_kwargs(cfg))

    model(torch.randint(0, cfg.vocab_size, (1, 4)), kv_cache=cache)
    assert cache.get_pos() == 4

    model(torch.randint(0, cfg.vocab_size, (1, 1)), kv_cache=cache)
    assert cache.get_pos() == 5
