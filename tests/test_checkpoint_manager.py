"""
Tests for checkpoint management.

Run with:
python -m pytest tests/test_checkpoint_manager.py -v -s
"""

import os
import tempfile
import pytest
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        n_embd=32,
    )
    model = GPT(config)
    model.init_weights()
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_save_checkpoint(tiny_model, temp_dir):
    """Test saving a checkpoint."""
    model = tiny_model
    
    # Prepare data
    model_data = model.state_dict()
    optimizer_data = {"step": 100}  # Mock optimizer data
    meta_data = {
        "iteration": 100,
        "model_config": model.config.__dict__,
        "train_config": {"lr": 0.001}
    }
    
    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=temp_dir,
        step=100,
        model_data=model_data,
        optimizer_data=optimizer_data,
        meta_data=meta_data
    )
    
    # Check that checkpoint files exist
    assert os.path.exists(os.path.join(temp_dir, "model_000100.pt"))
    assert os.path.exists(os.path.join(temp_dir, "optim_000100.pt"))
    assert os.path.exists(os.path.join(temp_dir, "meta_000100.json"))


def test_load_checkpoint(tiny_model, temp_dir):
    """Test loading a checkpoint."""
    model = tiny_model
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Prepare and save checkpoint
    model_data = model.state_dict()
    meta_data = {
        "iteration": 100,
        "model_config": model.config.__dict__,
    }
    
    save_checkpoint(
        checkpoint_dir=temp_dir,
        step=100,
        model_data=model_data,
        optimizer_data=None,
        meta_data=meta_data
    )
    
    # Load checkpoint back
    loaded_model_data, loaded_opt_data, loaded_meta = load_checkpoint(
        checkpoint_dir=temp_dir,
        step=100,
        device="cpu",
        load_optimizer=False
    )
    
    # Check that data matches
    for name in original_state:
        torch.testing.assert_close(loaded_model_data[name], original_state[name])
    
    # Check metadata
    assert loaded_meta['iteration'] == 100


def test_checkpoint_with_optimizer(tiny_model, temp_dir):
    """Test saving and loading with optimizer data."""
    model = tiny_model
    
    # Prepare checkpoint with optimizer
    model_data = model.state_dict()
    optimizer_data = {"step": 50, "lr": 0.001}
    meta_data = {"iteration": 50}
    
    save_checkpoint(
        checkpoint_dir=temp_dir,
        step=50,
        model_data=model_data,
        optimizer_data=optimizer_data,
        meta_data=meta_data
    )
    
    # Load with optimizer
    loaded_model, loaded_opt, loaded_meta = load_checkpoint(
        checkpoint_dir=temp_dir,
        step=50,
        device="cpu",
        load_optimizer=True
    )
    
    # Check optimizer data
    assert loaded_opt is not None
    assert loaded_opt["step"] == 50
    assert loaded_opt["lr"] == 0.001


def test_checkpoint_without_optimizer(tiny_model, temp_dir):
    """Test loading without optimizer data."""
    model = tiny_model
    
    # Save checkpoint without optimizer
    save_checkpoint(
        checkpoint_dir=temp_dir,
        step=75,
        model_data=model.state_dict(),
        optimizer_data=None,
        meta_data={"iteration": 75}
    )
    
    # Should not have optimizer file
    assert not os.path.exists(os.path.join(temp_dir, "optim_000075.pt"))
    
    # Load without optimizer should work
    loaded_model, loaded_opt, loaded_meta = load_checkpoint(
        checkpoint_dir=temp_dir,
        step=75,
        device="cpu",
        load_optimizer=False
    )
    
    assert loaded_opt is None


def test_checkpoint_metadata_preservation(tiny_model, temp_dir):
    """Test that metadata is preserved correctly."""
    model = tiny_model
    
    meta_data = {
        "iteration": 200,
        "model_config": model.config.__dict__,
        "train_config": {
            "lr": 0.02,
            "batch_size": 32,
            "max_iterations": 1000
        }
    }
    
    save_checkpoint(
        checkpoint_dir=temp_dir,
        step=200,
        model_data=model.state_dict(),
        optimizer_data=None,
        meta_data=meta_data
    )
    
    # Load and check metadata
    _, _, loaded_meta = load_checkpoint(
        checkpoint_dir=temp_dir,
        step=200,
        device="cpu"
    )
    
    assert loaded_meta['iteration'] == 200
    assert loaded_meta['train_config']['lr'] == 0.02
    assert loaded_meta['train_config']['batch_size'] == 32
