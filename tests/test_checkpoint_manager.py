
import os
import pytest
import json
import torch
from unittest.mock import MagicMock, patch
from nanochat.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    find_largest_model,
    find_last_step,
    load_model_from_dir,
    build_model
)

@pytest.fixture
def mock_dir(tmp_path):
    return tmp_path

def test_save_checkpoint(mock_dir):
    step = 100
    model_data = {"weights": torch.tensor([1.0])}
    optim_data = {"state": 1}
    meta_data = {"config": "test"}

    save_checkpoint(str(mock_dir), step, model_data, optim_data, meta_data, rank=0)

    assert (mock_dir / f"model_{step:06d}.pt").exists()
    assert (mock_dir / f"meta_{step:06d}.json").exists()
    assert (mock_dir / f"optim_{step:06d}_rank0.pt").exists()

def test_load_checkpoint(mock_dir):
    step = 100
    model_path = mock_dir / f"model_{step:06d}.pt"
    meta_path = mock_dir / f"meta_{step:06d}.json"
    optim_path = mock_dir / f"optim_{step:06d}_rank0.pt"

    torch.save({"weights": torch.tensor([1.0])}, model_path)
    torch.save({"state": 1}, optim_path)
    with open(meta_path, "w") as f:
        json.dump({"config": "test"}, f)

    m, o, meta = load_checkpoint(str(mock_dir), step, device=torch.device("cpu"), load_optimizer=True, rank=0)

    assert "weights" in m
    assert "state" in o
    assert meta["config"] == "test"

def test_find_largest_model(mock_dir):
    # create d12, d24 directories
    (mock_dir / "d12").mkdir()
    (mock_dir / "d24").mkdir()
    (mock_dir / "other").mkdir()

    tag = find_largest_model(str(mock_dir))
    assert tag == "d24"

def test_find_last_step(mock_dir):
    # create model_000100.pt, model_000200.pt
    (mock_dir / "model_000100.pt").touch()
    (mock_dir / "model_000200.pt").touch()

    step = find_last_step(str(mock_dir))
    assert step == 200

@patch("nanochat.checkpoint_manager.load_checkpoint")
@patch("nanochat.checkpoint_manager.get_tokenizer")
@patch("nanochat.checkpoint_manager.GPT")
def test_build_model(mock_gpt, mock_get_tokenizer, mock_load_checkpoint, mock_dir):
    # Setup mocks
    mock_load_checkpoint.return_value = ({}, None, {"model_config": {"vocab_size": 100}})

    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab_size.return_value = 100
    mock_get_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_gpt.return_value = mock_model

    device = torch.device("cpu")
    model, tokenizer, meta = build_model(str(mock_dir), 100, device, phase="eval")

    assert model == mock_model
    assert tokenizer == mock_tokenizer
    assert meta["model_config"]["vocab_size"] == 100
    mock_model.eval.assert_called_once()
    mock_model.load_state_dict.assert_called_once()

def test_load_model_from_dir_guesses(mock_dir):
    # Test that it guesses tag and step correctly
    (mock_dir / "d10").mkdir()
    checkpoint_dir = mock_dir / "d10"
    (checkpoint_dir / "model_000050.pt").touch()
    (checkpoint_dir / "meta_000050.json").touch() # needed for load_checkpoint mock if we weren't mocking build_model

    with patch("nanochat.checkpoint_manager.build_model") as mock_build:
        mock_build.return_value = ("model", "tok", "meta")

        m, t, meta = load_model_from_dir(str(mock_dir), torch.device("cpu"), "eval")

        assert m == "model"
        # Verify it called build_model with correct paths
        args, _ = mock_build.call_args
        assert str(checkpoint_dir) in args[0]
        assert args[1] == 50
