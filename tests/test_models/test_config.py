"""Tests for GPT model configuration."""
from nanochat.models import GPTConfig


def test_gpt_config_creation():
    """Test GPTConfig can be created with valid parameters."""
    config = GPTConfig(
        sequence_len=1024,
        vocab_size=32768,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,
    )
    assert config.sequence_len == 1024
    assert config.vocab_size == 32768
    assert config.n_layer == 12
    assert config.n_embd == 768
    assert config.n_head == 12
