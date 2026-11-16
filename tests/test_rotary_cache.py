"""RoPE cache must grow when few-shot eval prompts exceed sequence_len * 10."""

import torch

from nanochat.gpt import GPT, GPTConfig


def _tiny_model(sequence_len: int) -> GPT:
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=64,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=64,
        window_pattern="L",
    )
    return GPT(config)


def test_rotary_cache_starts_at_ten_times_sequence_len():
    model = _tiny_model(sequence_len=8)
    assert model.cos.size(1) == 80


def test_forward_grows_rotary_cache_past_initial_limit():
    model = _tiny_model(sequence_len=8)
    initial = model.cos.size(1)
    # Mimic the live failure shape: T just past sequence_len * 10 (256 → 2560 vs 2596).
    T = initial + 36
    idx = torch.randint(0, model.config.vocab_size, (1, T))
    logits = model(idx)
    assert logits.shape == (1, T, model.config.vocab_size)
    # Grow with a 2x buffer so the next slightly-longer prompt does not reallocate.
    assert model.cos.size(1) == max(T, initial * 2)
    assert model.rotary_seq_len == model.cos.size(1)
