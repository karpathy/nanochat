"""Test optimizer setup."""
import torch
from nanochat.models import GPT, GPTConfig


def test_optimizer_setup():
    """Test optimizer can be configured."""
    config = GPTConfig(
        sequence_len=512,
        vocab_size=1000,
        n_layer=4,
        n_embd=256,
        n_head=4,
        n_kv_head=4,
    )
    model = GPT(config)
    model.init_weights()
    
    optimizer = model.setup_optimizer(
        matrix_lr=0.02,
        embedding_lr=0.3,
        unembedding_lr=0.008,
        scalar_lr=0.5,
        weight_decay=0.28,
    )
    
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0


def test_optimizer_param_groups():
    """Test optimizer has correct parameter groups."""
    config = GPTConfig(
        sequence_len=256,
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
        weight_decay=0.28,
    )
    
    # Check that we have multiple param groups (Muon + Adam)
    assert len(optimizer.param_groups) >= 2
    
    # Check that each group has required keys
    for group in optimizer.param_groups:
        assert 'lr' in group
        assert 'params' in group
        assert len(group['params']) > 0


def test_optimizer_step():
    """Test optimizer can perform a step."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_embd=64,
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
    
    # Forward pass
    inputs = torch.randint(0, 100, (2, 64))
    targets = torch.randint(0, 100, (2, 64))
    loss = model(inputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Check that loss is still valid after step
    loss2 = model(inputs, targets)
    assert not torch.isnan(loss2)
