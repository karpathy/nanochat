"""Test model configuration."""
from nanochat.models import GPTConfig, TrainingConfig


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


def test_training_config_creation():
    """Test TrainingConfig can be created."""
    config = TrainingConfig(
        depth=12,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        num_iterations=1000,
        device_batch_size=32,
        total_batch_size=524288,
    )
    assert config.depth == 12
    assert config.max_seq_len == 2048
    assert config.num_iterations == 1000
