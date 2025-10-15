"""
Tests for hybrid block architecture and backward compatibility.
"""

import pytest
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.blocks import BaseBlock, create_block
from nanochat.blocks.transformer_block import TransformerBlock
from nanochat.blocks.mamba_block import MambaBlock


def test_backward_compatibility_default_config():
    """Test that default config (no block_pattern) creates all transformer blocks."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    
    with torch.device("meta"):
        model = GPT(config)
    
    # Check that all blocks are transformer blocks
    for i, block in enumerate(model.transformer.h):
        assert hasattr(block, 'attn'), f"Block {i} should be TransformerBlock with 'attn' attribute"
        assert hasattr(block, 'mlp'), f"Block {i} should have 'mlp' attribute"


def test_explicit_transformer_pattern():
    """Test explicit all-transformer pattern matches default."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        block_pattern=["T", "T", "T", "T"],
    )
    
    with torch.device("meta"):
        model = GPT(config)
    
    # Check that all blocks are transformer blocks
    for i, block in enumerate(model.transformer.h):
        assert hasattr(block, 'attn'), f"Block {i} should be TransformerBlock"


def test_hybrid_pattern():
    """Test that hybrid patterns create correct block types."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        block_pattern=["T", "T", "M", "M"],
        mamba_d_state=16,
    )
    
    with torch.device("meta"):
        model = GPT(config)
    
    # Check block types
    assert hasattr(model.transformer.h[0], 'attn'), "Block 0 should be TransformerBlock"
    assert hasattr(model.transformer.h[1], 'attn'), "Block 1 should be TransformerBlock"
    assert hasattr(model.transformer.h[2], 'mixer'), "Block 2 should be MambaBlock"
    assert hasattr(model.transformer.h[3], 'mixer'), "Block 3 should be MambaBlock"


def test_alternating_pattern():
    """Test alternating transformer-mamba pattern."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=6,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        block_pattern=["T", "M", "T", "M", "T", "M"],
        mamba_d_state=16,
    )
    
    with torch.device("meta"):
        model = GPT(config)
    
    # Check alternating pattern
    for i, block in enumerate(model.transformer.h):
        if i % 2 == 0:
            assert hasattr(block, 'attn'), f"Block {i} should be TransformerBlock"
        else:
            assert hasattr(block, 'mixer'), f"Block {i} should be MambaBlock"


def test_block_pattern_validation():
    """Test that invalid block patterns raise errors."""
    # Wrong length
    with pytest.raises(ValueError, match="must match"):
        config = GPTConfig(
            n_layer=4,
            block_pattern=["T", "T"],  # Only 2 but n_layer=4
        )
        with torch.device("meta"):
            model = GPT(config)
    
    # Invalid block type
    with pytest.raises(ValueError, match="Unknown block type"):
        config = GPTConfig(
            n_layer=2,
            block_pattern=["T", "X"],  # X is invalid
        )
        with torch.device("meta"):
            model = GPT(config)


def test_block_factory():
    """Test the block factory function."""
    config = GPTConfig(n_embd=128, n_head=2, n_kv_head=2)
    
    # Test transformer block creation
    block_t = create_block("T", config, 0)
    assert isinstance(block_t, BaseBlock)
    assert hasattr(block_t, 'attn')
    
    block_transformer = create_block("transformer", config, 0)
    assert isinstance(block_transformer, BaseBlock)
    assert hasattr(block_transformer, 'attn')
    
    # Test mamba block creation
    block_m = create_block("M", config, 0)
    assert isinstance(block_m, BaseBlock)
    assert hasattr(block_m, 'mixer')
    
    block_mamba = create_block("mamba", config, 0)
    assert isinstance(block_mamba, BaseBlock)
    assert hasattr(block_mamba, 'mixer')


def test_forward_pass_transformer():
    """Test forward pass through pure transformer model."""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        block_pattern=["T", "T"],
    )
    
    model = GPT(config)
    model.init_weights()
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, 1000)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for this test"
)
def test_forward_pass_hybrid_gpu():
    """Test forward pass through hybrid model on GPU (requires mamba-ssm)."""
    try:
        import mamba_ssm
    except ImportError:
        pytest.skip("mamba-ssm not installed")
    
    config = GPTConfig(
        sequence_len=32,
        vocab_size=1000,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        block_pattern=["T", "M", "T", "M"],
        mamba_d_state=8,
    )
    
    device = torch.device("cuda")
    model = GPT(config).to(device)
    model.init_weights()
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, 1000)
    assert logits.device.type == "cuda"


def test_model_config_serialization():
    """Test that model config with block_pattern can be serialized."""
    import json
    
    config = GPTConfig(
        n_layer=4,
        block_pattern=["T", "T", "M", "M"],
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    )
    
    # Convert to dict (as done in checkpoint_manager)
    config_dict = {
        "sequence_len": config.sequence_len,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_embd": config.n_embd,
        "block_pattern": config.block_pattern,
        "mamba_d_state": config.mamba_d_state,
        "mamba_d_conv": config.mamba_d_conv,
        "mamba_expand": config.mamba_expand,
        "mamba_use_mlp": config.mamba_use_mlp,
    }
    
    # Should be JSON serializable
    json_str = json.dumps(config_dict)
    loaded = json.loads(json_str)
    
    # Reconstruct config
    new_config = GPTConfig(**loaded)
    assert new_config.block_pattern == config.block_pattern
    assert new_config.mamba_d_state == config.mamba_d_state


def test_parameter_count_consistency():
    """Test that transformer and mamba blocks have similar parameter counts."""
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    )
    
    # Create one transformer block
    transformer_block = create_block("T", config, 0)
    transformer_params = transformer_block.get_num_params()
    
    # Create one mamba block
    mamba_block = create_block("M", config, 0)
    mamba_params = mamba_block.get_num_params()
    
    # Should be roughly similar (within 2x)
    ratio = max(transformer_params, mamba_params) / min(transformer_params, mamba_params)
    assert ratio < 2.0, f"Parameter count ratio too large: {ratio:.2f}"


if __name__ == "__main__":
    # Run basic tests
    print("Running backward compatibility tests...")
    test_backward_compatibility_default_config()
    print("✓ Default config creates transformer blocks")
    
    test_explicit_transformer_pattern()
    print("✓ Explicit transformer pattern works")
    
    test_hybrid_pattern()
    print("✓ Hybrid pattern creates correct block types")
    
    test_alternating_pattern()
    print("✓ Alternating pattern works")
    
    test_block_factory()
    print("✓ Block factory works")
    
    test_forward_pass_transformer()
    print("✓ Forward pass works for transformer")
    
    test_model_config_serialization()
    print("✓ Config serialization works")
    
    print("\nAll tests passed! ✓")

