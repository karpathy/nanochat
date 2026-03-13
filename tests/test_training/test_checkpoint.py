"""Test checkpoint save/load."""
import os
import tempfile
import torch
from nanochat.models import GPT, GPTConfig
from nanochat.training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_save_load():
    """Test checkpoint can be saved and loaded."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=500,
        n_layer=2,
        n_embd=128,
        n_head=2,
        n_kv_head=2,
    )
    model = GPT(config)
    model.init_weights()
    
    optimizer = model.setup_optimizer(
        matrix_lr=0.02,
        embedding_lr=0.3,
        unembedding_lr=0.008,
        scalar_lr=0.5,
        weight_decay=0.0,
    )
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            tmpdir,
            step=100,
            model_data=model.state_dict(),
            optimizer_data=optimizer.state_dict(),
            meta_data={"test": "value", "step": 100},
            rank=0,
        )
        
        # Check files exist
        assert os.path.exists(os.path.join(tmpdir, "model_000100.pt"))
        assert os.path.exists(os.path.join(tmpdir, "optim_000100_rank0.pt"))
        assert os.path.exists(os.path.join(tmpdir, "meta_000100.json"))
        
        # Load checkpoint
        model_data, optimizer_data, meta_data = load_checkpoint(
            tmpdir,
            step=100,
            device=torch.device("cpu"),
            load_optimizer=True,
            rank=0,
        )
        
        assert model_data is not None
        assert optimizer_data is not None
        assert meta_data is not None
        assert meta_data["test"] == "value"
        assert meta_data["step"] == 100


def test_checkpoint_model_restore():
    """Test model weights are correctly restored from checkpoint."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=200,
        n_layer=2,
        n_embd=64,
        n_head=2,
        n_kv_head=2,
    )
    model1 = GPT(config)
    model1.init_weights()
    
    # Get initial weights from first layer
    initial_weight = model1.transformer.h[0].attn.c_q.weight.clone()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_checkpoint(
            tmpdir,
            step=50,
            model_data=model1.state_dict(),
            optimizer_data={},
            meta_data={"step": 50},
            rank=0,
        )
        
        # Create new model and load
        model2 = GPT(config)
        model2.init_weights()  # Initialize with different weights
        
        model_data, _, _ = load_checkpoint(
            tmpdir,
            step=50,
            device=torch.device("cpu"),
            load_optimizer=False,
            rank=0,
        )
        
        model2.load_state_dict(model_data)
        
        # Check weights match
        restored_weight = model2.transformer.h[0].attn.c_q.weight
        assert torch.allclose(initial_weight, restored_weight)
