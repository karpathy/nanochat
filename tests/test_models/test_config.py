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


def test_training_config_save_load_roundtrip(tmp_path):
    """Test save() then load() produces identical config."""
    config = TrainingConfig(
        depth=16,
        aspect_ratio=48,
        max_seq_len=1024,
        num_iterations=500,
        device_batch_size=8,
        track_compression=True,
        compression_log_every=50,
    )
    path = tmp_path / "config.toml"
    config.save(path)
    loaded = TrainingConfig.load(path)
    assert loaded.depth == config.depth
    assert loaded.aspect_ratio == config.aspect_ratio
    assert loaded.max_seq_len == config.max_seq_len
    assert loaded.num_iterations == config.num_iterations
    assert loaded.device_batch_size == config.device_batch_size
    assert loaded.track_compression == config.track_compression
    assert loaded.compression_log_every == config.compression_log_every


def test_training_config_from_args():
    """Test from_args() maps argparse Namespace to TrainingConfig."""
    from argparse import Namespace

    args = Namespace(
        depth=20,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        num_iterations=100,
        device_batch_size=4,
        total_batch_size=524288,
        run="test",
        base_dir="/tmp/nanochat",
        unrelated_arg="ignored",
    )
    config = TrainingConfig.from_args(args)
    assert config.depth == 20
    assert config.device_batch_size == 4
    assert config.run == "test"
    assert config.base_dir == "/tmp/nanochat"
    assert not hasattr(config, "unrelated_arg")


def test_training_config_base_dir_default():
    """Test base_dir defaults to None, which resolves to ~/.cache/nanochat/."""
    import os
    from nanochat.common import get_base_dir

    config = TrainingConfig(depth=12)
    assert config.base_dir is None

    # When base_dir is None, get_base_dir() resolves to ~/.cache/nanochat/
    env_backup = os.environ.pop("NANOCHAT_BASE_DIR", None)
    try:
        expected = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
        assert get_base_dir() == expected
    finally:
        if env_backup is not None:
            os.environ["NANOCHAT_BASE_DIR"] = env_backup
