"""Checkpoint-missing errors should be actionable. python -m pytest tests/test_checkpoint_errors.py -v"""
import pytest
import torch
from nanochat.checkpoint_manager import load_model


def test_missing_base_checkpoints_message(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as e:
        load_model("base", torch.device("cpu"), phase="eval")
    msg = str(e.value)
    assert str(tmp_path / "base_checkpoints") in msg
    assert "'base'" in msg and "base_train" in msg and "RUNNING.md" in msg


def test_missing_sft_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="chat_sft"):
        load_model("sft", torch.device("cpu"), phase="eval")
