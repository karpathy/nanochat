"""Tests for nanochat.common.config."""

import tomllib
from pathlib import Path

import pytest

from nanochat.config import (
    CommonConfig,
    Config,
    ConfigLoader,
    EvaluationConfig,
    RLConfig,
    SFTConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_common_config_defaults():
    cfg = CommonConfig()
    assert cfg.base_dir is None
    assert cfg.device_type == ""
    assert cfg.run == "unnamed"
    assert cfg.wandb == "local"
    assert cfg.wandb_project == "nanochat"


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.depth == 20
    assert cfg.aspect_ratio == 64
    assert cfg.fp8 is False
    assert cfg.model_tag is None


def test_sft_config_defaults():
    cfg = SFTConfig()
    assert cfg.load_optimizer is True
    assert cfg.num_iterations == -1
    assert cfg.model_tag is None


def test_rl_config_defaults():
    cfg = RLConfig()
    assert cfg.num_epochs == 1
    assert cfg.temperature == 1.0


def test_evaluation_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.modes == "core,bpb,sample"
    assert cfg.max_per_task == -1


def test_evaluation_config_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid eval modes"):
        EvaluationConfig(modes="core,bogus")


def test_evaluation_config_valid_subset():
    cfg = EvaluationConfig(modes="core,bpb")
    assert cfg.modes == "core,bpb"


# ---------------------------------------------------------------------------
# generate_default — each section produces valid TOML
# ---------------------------------------------------------------------------


def test_common_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[common]\n" + CommonConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert "common" in data


def test_training_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text("[training]\n" + TrainingConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["training"]["depth"] == 20


def test_sft_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text("[sft]\n" + SFTConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["sft"]["mmlu_epochs"] == 3


def test_rl_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "r.toml"
    p.write_text("[rl]\n" + RLConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["rl"]["num_epochs"] == 1


def test_evaluation_generate_default_is_valid_toml(tmp_path):
    p = tmp_path / "e.toml"
    p.write_text("[evaluation]\n" + EvaluationConfig.generate_default(), encoding="utf-8")
    data = tomllib.loads(p.read_text())
    assert data["evaluation"]["device_batch_size"] == 32


def test_config_generate_default_all_sections():
    text = Config.generate_default()
    data = tomllib.loads(text)
    for section in ("common", "training", "sft", "rl", "evaluation"):
        assert section in data, f"missing [{section}]"
    assert data["training"]["depth"] == 20
    assert data["sft"]["mmlu_epochs"] == 3
    assert data["rl"]["num_epochs"] == 1
    assert data["evaluation"]["device_batch_size"] == 32


def test_config_load(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[common]\nrun = \"loaded\"\n[training]\ndepth = 7\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.common.run == "loaded"
    assert cfg.training.depth == 7
    assert cfg.sft.mmlu_epochs == 3  # default intact


def test_config_load_unknown_section_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[bogus]\nfoo = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        Config.load(p)

def test_save_roundtrip(tmp_path):
    cfg = Config()
    cfg.common.run = "roundtrip"
    cfg.training.depth = 8
    cfg.sft.mmlu_epochs = 5
    cfg.rl.num_epochs = 3
    cfg.evaluation.max_per_task = 100
    p = tmp_path / "config.toml"
    cfg.save(p)

    with open(p, "rb") as f:
        data = tomllib.load(f)

    assert data["common"]["run"] == "roundtrip"
    assert data["training"]["depth"] == 8
    assert data["sft"]["mmlu_epochs"] == 5
    assert data["rl"]["num_epochs"] == 3
    assert data["evaluation"]["max_per_task"] == 100


def test_save_omits_none_fields(tmp_path):
    cfg = Config()
    assert cfg.training.model_tag is None
    p = tmp_path / "config.toml"
    cfg.save(p)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    assert "model_tag" not in data["training"]


# ---------------------------------------------------------------------------
# ConfigLoader — CLI only (no TOML)
# ---------------------------------------------------------------------------


def test_parse_cli_only_common(tmp_path):
    cfg = ConfigLoader().parse(["--base-dir", str(tmp_path), "--run", "smoke", "--wandb", "disabled"])
    assert cfg.common.run == "smoke"
    assert cfg.common.wandb == "disabled"


def test_parse_cli_only_training(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path), "--depth", "6", "--num-iterations", "20"])
    assert cfg.training.depth == 6
    assert cfg.training.num_iterations == 20


def test_parse_cli_only_sft(tmp_path):
    cfg = ConfigLoader().add_sft().parse(["--base-dir", str(tmp_path), "--mmlu-epochs", "7", "--no-load-optimizer"])
    assert cfg.sft.mmlu_epochs == 7
    assert cfg.sft.load_optimizer is False


def test_parse_cli_only_rl(tmp_path):
    cfg = ConfigLoader().add_rl().parse(["--base-dir", str(tmp_path), "--num-epochs", "4", "--temperature", "0.8"])
    assert cfg.rl.num_epochs == 4
    assert cfg.rl.temperature == 0.8


def test_parse_cli_only_evaluation(tmp_path):
    cfg = ConfigLoader().add_evaluation().parse(["--base-dir", str(tmp_path), "--max-per-task", "50"])
    assert cfg.evaluation.max_per_task == 50


def test_parse_no_args_uses_defaults(tmp_path):
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.training.depth == 20
    assert cfg.common.run == "unnamed"


# ---------------------------------------------------------------------------
# ConfigLoader — TOML only (no CLI overrides)
# ---------------------------------------------------------------------------


def test_parse_toml_common(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[common]\nrun = \"from-toml\"\n", encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p)])
    assert cfg.common.run == "from-toml"


def test_parse_toml_training(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 12\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg.training.depth == 12


def test_parse_toml_sft(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[sft]\nmmlu_epochs = 9\n", encoding="utf-8")
    cfg = ConfigLoader().add_sft().parse(["--config", str(p)])
    assert cfg.sft.mmlu_epochs == 9


def test_parse_toml_rl(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[rl]\nnum_epochs = 5\n", encoding="utf-8")
    cfg = ConfigLoader().add_rl().parse(["--config", str(p)])
    assert cfg.rl.num_epochs == 5


def test_parse_toml_evaluation(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[evaluation]\nmax_per_task = 200\n", encoding="utf-8")
    cfg = ConfigLoader().add_evaluation().parse(["--config", str(p)])
    assert cfg.evaluation.max_per_task == 200


# ---------------------------------------------------------------------------
# ConfigLoader — CLI overrides TOML
# ---------------------------------------------------------------------------


def test_cli_overrides_toml_training(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 12\nnum_iterations = 1000\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p), "--num-iterations", "500"])
    assert cfg.training.depth == 12          # TOML, not touched by CLI
    assert cfg.training.num_iterations == 500  # CLI wins


def test_cli_overrides_toml_common(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[common]\nrun = \"toml-run\"\n", encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p), "--run", "cli-run"])
    assert cfg.common.run == "cli-run"


def test_toml_beats_dataclass_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 99\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg.training.depth == 99


# ---------------------------------------------------------------------------
# ConfigLoader — base_dir autodiscovery
# ---------------------------------------------------------------------------


def test_base_dir_autodiscovery(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[common]\nrun = \"discovered\"\n[training]\ndepth = 3\n", encoding="utf-8")
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.common.run == "discovered"
    assert cfg.training.depth == 3


def test_explicit_config_overrides_base_dir(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.toml").write_text("[common]\nrun = \"base\"\n", encoding="utf-8")
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[common]\nrun = \"explicit\"\n", encoding="utf-8")
    cfg = ConfigLoader().parse(["--base-dir", str(base), "--config", str(explicit)])
    assert cfg.common.run == "explicit"


def test_base_dir_no_config_toml_uses_defaults(tmp_path):
    # base_dir exists but has no config.toml — should not error, use defaults
    cfg = ConfigLoader().add_training().parse(["--base-dir", str(tmp_path)])
    assert cfg.training.depth == 20


def test_multiple_sections_raises():
    import pytest
    with pytest.raises(RuntimeError, match="one section"):
        ConfigLoader().add_training().add_sft()




def test_unregistered_section_in_toml_ignored(tmp_path):
    # training section present in TOML but add_training() not called — section skipped
    p = tmp_path / "config.toml"
    p.write_text("[training]\ndepth = 5\n", encoding="utf-8")
    cfg = ConfigLoader().parse(["--config", str(p)])  # no add_training()
    assert cfg.training.depth == 20  # default, TOML section was not applied


# ---------------------------------------------------------------------------
# ConfigLoader — save then parse roundtrip
# ---------------------------------------------------------------------------


def test_save_then_parse_roundtrip(tmp_path):
    cfg = Config()
    cfg.common.run = "roundtrip"
    cfg.common.wandb = "disabled"
    cfg.training.depth = 4
    cfg.training.num_iterations = 300
    p = tmp_path / "config.toml"
    cfg.save(p)

    cfg2 = ConfigLoader().add_training().parse(["--config", str(p)])
    assert cfg2.common.run == "roundtrip"
    assert cfg2.common.wandb == "disabled"
    assert cfg2.training.depth == 4
    assert cfg2.training.num_iterations == 300
