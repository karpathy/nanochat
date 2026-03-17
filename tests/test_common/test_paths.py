"""Test paths module."""

import os
from nanochat.common.paths import (
    checkpoint_dir,
    data_dir,
    eval_results_dir,
    eval_tasks_dir,
    identity_data_path,
    legacy_data_dir,
    tokenizer_dir,
)


def test_data_dir(tmp_path):
    assert data_dir(str(tmp_path)) == str(tmp_path / "data" / "climbmix")
    assert os.path.isdir(data_dir(str(tmp_path)))


def test_legacy_data_dir(tmp_path):
    result = legacy_data_dir(str(tmp_path))
    assert result == str(tmp_path / "data" / "fineweb")
    assert not os.path.isdir(result)  # not auto-created


def test_tokenizer_dir(tmp_path):
    assert tokenizer_dir(str(tmp_path)) == str(tmp_path / "tokenizer")
    assert os.path.isdir(tokenizer_dir(str(tmp_path)))


def test_checkpoint_dir(tmp_path):
    base = str(tmp_path)
    assert checkpoint_dir(base, "base", "d12") == str(tmp_path / "checkpoints" / "base" / "d12")
    assert checkpoint_dir(base, "sft", "d20") == str(tmp_path / "checkpoints" / "sft" / "d20")
    assert checkpoint_dir(base, "rl", "d24") == str(tmp_path / "checkpoints" / "rl" / "d24")
    assert os.path.isdir(checkpoint_dir(base, "base", "d12"))


def test_checkpoints_dir(tmp_path):
    assert checkpoint_dir(str(tmp_path), "base") == str(tmp_path / "checkpoints" / "base")


def test_eval_dirs(tmp_path):
    base = str(tmp_path)
    assert eval_tasks_dir(base) == str(tmp_path / "data" / "eval_tasks")
    assert eval_results_dir(base) == str(tmp_path / "eval")


def test_identity_data_path(tmp_path):
    assert identity_data_path(str(tmp_path)) == str(tmp_path / "identity.jsonl")


def test_base_dir_override(tmp_path):
    custom = str(tmp_path / "custom")
    os.makedirs(custom)
    assert data_dir(custom) == str(tmp_path / "custom" / "data" / "climbmix")
    assert tokenizer_dir(custom) == str(tmp_path / "custom" / "tokenizer")
    assert checkpoint_dir(custom, "base", "d12") == str(tmp_path / "custom" / "checkpoints" / "base" / "d12")
