"""Tests for nanochat.common.config — Config root class and argparse builders."""

import argparse
import tomllib

from nanochat.common.config import (
    Config,
    add_common_args,
    add_training_args,
)


def test_load_empty_toml(tmp_path):
    """Config.load() with only [common] populates section defaults."""
    p = tmp_path / "config.toml"
    p.write_text("[common]\nrun = \"smoke\"\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.common.run == "smoke"
    # section defaults are intact
    assert cfg.training.depth == 20
    assert cfg.sft.mmlu_epochs == 3
    assert cfg.rl.num_epochs == 1
    assert cfg.evaluation.modes == "core,bpb,sample"


def test_common_fields_inherited(tmp_path):
    """Section fields inherit [common] values on load."""
    p = tmp_path / "config.toml"
    p.write_text(
        "[common]\nrun = \"my-run\"\nwandb = \"disabled\"\n",
        encoding="utf-8",
    )
    cfg = Config.load(p)
    assert cfg.training.run == "my-run"
    assert cfg.training.wandb == "disabled"
    assert cfg.sft.run == "my-run"
    assert cfg.sft.wandb == "disabled"
    assert cfg.rl.wandb == "disabled"
    assert cfg.evaluation.wandb == "disabled"


def test_cli_overrides_toml(tmp_path):
    """apply_args() with SUPPRESS only overrides explicitly-passed args."""
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 12\nnum_iterations = 1000\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.training.depth == 12
    assert cfg.training.num_iterations == 1000

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_training_args(parser)
    explicit = parser.parse_args(["--num-iterations", "500"])
    cfg.apply_args(explicit, "training")

    assert cfg.training.depth == 12          # not touched — not on CLI
    assert cfg.training.num_iterations == 500  # overridden


def test_from_args_no_config():
    """Config.from_args() builds correct config without a TOML file."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_training_args(parser)
    # parse_args with explicit list — no SUPPRESS, so defaults are present
    args = parser.parse_args(["--depth", "6", "--run", "smoke", "--num-iterations", "20"])
    cfg = Config.from_args(args, "training")
    assert cfg.training.depth == 6
    assert cfg.common.run == "smoke"
    assert cfg.training.num_iterations == 20


def test_save_roundtrip(tmp_path):
    """save() then load() produces identical Config."""
    p = tmp_path / "config.toml"
    cfg = Config()
    cfg.common.run = "roundtrip"
    cfg.common.wandb = "disabled"
    cfg.training.depth = 8
    cfg.training.num_iterations = 200
    cfg.sft.mmlu_epochs = 5
    cfg.rl.num_epochs = 3
    cfg.evaluation.max_per_task = 100
    cfg.save(p)

    cfg2 = Config.load(p)
    assert cfg2.common.run == "roundtrip"
    assert cfg2.common.wandb == "disabled"
    assert cfg2.training.depth == 8
    assert cfg2.training.num_iterations == 200
    assert cfg2.sft.mmlu_epochs == 5
    assert cfg2.rl.num_epochs == 3
    assert cfg2.evaluation.max_per_task == 100


def test_save_omits_common_fields(tmp_path):
    """Saved TOML does not duplicate common fields inside sub-sections."""
    p = tmp_path / "config.toml"
    cfg = Config()
    cfg.common.run = "check"
    cfg.training.depth = 4
    cfg.save(p)

    with open(p, "rb") as f:
        data = tomllib.load(f)

    # common fields must NOT appear inside [training]
    training = data.get("training", {})
    assert "run" not in training
    assert "wandb" not in training
    assert "base_dir" not in training
    assert "device_type" not in training
    # but the section-specific field is there
    assert training.get("depth") == 4


def test_base_dir_autodiscovery(tmp_path):
    """Config.load() from base_dir/config.toml picks up the file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[common]\nrun = \"discovered\"\n[training]\ndepth = 3\n",
        encoding="utf-8",
    )
    cfg = Config.load(config_path)
    assert cfg.common.run == "discovered"
    assert cfg.training.depth == 3


def test_generate_default_is_valid_toml(tmp_path):
    """generate_default() output parses without error and has all sections."""
    p = tmp_path / "config.toml"
    Config.generate_default(p)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    for section in ("common", "training", "sft", "rl", "evaluation"):
        assert section in data, f"missing section [{section}]"
    # spot-check a few values match the dataclass defaults
    assert data["training"]["depth"] == 20
    assert data["sft"]["mmlu_epochs"] == 3
    assert data["rl"]["num_epochs"] == 1
    assert data["evaluation"]["device_batch_size"] == 32


# ---------------------------------------------------------------------------
# Migrated from test_models/test_config.py — now testing common.TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_creation():
    """TrainingConfig can be created with explicit fields."""
    from nanochat.common.config import TrainingConfig

    cfg = TrainingConfig(
        depth=12,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=2048,
        num_iterations=1000,
        device_batch_size=32,
        total_batch_size=524288,
    )
    assert cfg.depth == 12
    assert cfg.max_seq_len == 2048
    assert cfg.num_iterations == 1000


def test_training_config_save_load_roundtrip(tmp_path):
    """Config.save() then Config.load() preserves TrainingConfig fields."""
    cfg = Config()
    cfg.training.depth = 16
    cfg.training.aspect_ratio = 48
    cfg.training.max_seq_len = 1024
    cfg.training.num_iterations = 500
    cfg.training.device_batch_size = 8
    cfg.training.track_compression = True
    cfg.training.compression_log_every = 50

    p = tmp_path / "config.toml"
    cfg.save(p)
    loaded = Config.load(p)

    assert loaded.training.depth == 16
    assert loaded.training.aspect_ratio == 48
    assert loaded.training.max_seq_len == 1024
    assert loaded.training.num_iterations == 500
    assert loaded.training.device_batch_size == 8
    assert loaded.training.track_compression is True
    assert loaded.training.compression_log_every == 50


def test_training_config_from_args():
    """Config.from_args() maps argparse Namespace to TrainingConfig."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_training_args(parser)
    args = parser.parse_args([
        "--depth", "20",
        "--num-iterations", "100",
        "--device-batch-size", "4",
        "--run", "test",
        "--base-dir", "/tmp/nanochat",
    ])
    cfg = Config.from_args(args, "training")
    assert cfg.training.depth == 20
    assert cfg.training.device_batch_size == 4
    assert cfg.common.run == "test"
    assert cfg.common.base_dir == "/tmp/nanochat"
    assert not hasattr(cfg.training, "unrelated_arg")


def test_training_config_base_dir_default():
    """base_dir defaults to None; get_base_dir() resolves to ~/.cache/nanochat/."""
    import os
    from nanochat.common import get_base_dir
    from nanochat.common.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.base_dir is None

    env_backup = os.environ.pop("NANOCHAT_BASE_DIR", None)
    try:
        expected = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
        assert get_base_dir() == expected
    finally:
        if env_backup is not None:
            os.environ["NANOCHAT_BASE_DIR"] = env_backup


def test_section_override_beats_common(tmp_path):
    """A section-specific value overrides the [common] value for that field."""
    p = tmp_path / "config.toml"
    p.write_text(
        "[common]\nwandb = \"local\"\n[training]\nwandb = \"disabled\"\n",
        encoding="utf-8",
    )
    cfg = Config.load(p)
    assert cfg.common.wandb == "local"
    assert cfg.training.wandb == "disabled"   # section wins
    assert cfg.sft.wandb == "local"           # inherits common
