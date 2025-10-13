"""
Tests for common utility functions.

Run with:
python -m pytest tests/test_common.py -v -s --timeout=60
"""

import os
import pytest
import torch
import torch.distributed as dist
from nanochat.common import (
    get_base_dir,
    print0,
    is_ddp,
    get_dist_info,
    DummyWandb
)


def test_get_base_dir():
    """Test that base directory is created and returned."""
    base_dir = get_base_dir()
    
    # Should return a valid path
    assert isinstance(base_dir, str)
    assert len(base_dir) > 0
    
    # Directory should exist
    assert os.path.exists(base_dir)
    assert os.path.isdir(base_dir)
    
    # Should contain 'nanochat' in the path
    assert 'nanochat' in base_dir


def test_get_base_dir_custom():
    """Test custom base directory via environment variable."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = os.path.join(tmpdir, "custom_nanochat")
        
        # Set environment variable
        old_env = os.environ.get("NANOCHAT_BASE_DIR")
        os.environ["NANOCHAT_BASE_DIR"] = custom_dir
        
        try:
            base_dir = get_base_dir()
            
            # Should return custom directory
            assert base_dir == custom_dir
            assert os.path.exists(base_dir)
        finally:
            # Restore environment
            if old_env is None:
                os.environ.pop("NANOCHAT_BASE_DIR", None)
            else:
                os.environ["NANOCHAT_BASE_DIR"] = old_env


def test_print0(capsys):
    """Test print0 function."""
    # In non-DDP mode, should print
    print0("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out


def test_is_ddp():
    """Test DDP detection."""
    # In test environment, should not be DDP
    assert is_ddp() == False


def test_get_dist_info():
    """Test getting distributed info."""
    ddp, rank, local_rank, world_size = get_dist_info()
    
    # In test environment, should not be DDP
    assert ddp == False
    assert rank == 0
    assert local_rank == 0
    assert world_size == 1


def test_dummy_wandb():
    """Test DummyWandb class."""
    wandb = DummyWandb()
    
    # Should have log method
    assert hasattr(wandb, 'log')
    
    # Should have finish method
    assert hasattr(wandb, 'finish')
    
    # Methods should do nothing but not error
    wandb.log({"loss": 0.5})
    wandb.finish()


def test_dummy_wandb_kwargs():
    """Test DummyWandb accepts arbitrary kwargs."""
    wandb = DummyWandb()
    
    # Should accept any arguments without error
    wandb.log({"loss": 0.5}, step=10, commit=True)
    wandb.log(arbitrary_arg="value", another=123)

