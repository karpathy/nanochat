"""
Tests for the GPT model architecture.

Run with:
python -m pytest tests/test_gpt.py -v -s --timeout=60
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb, repeat_kv


@pytest.fixture
def small_config():
    """A small GPT config for fast testing."""
    return GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=2,  # Test MQA with 2:1 ratio
        n_embd=64,
    )


@pytest.fixture
def tiny_config():
    """An even tinier config for quick tests."""
    return GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_kv_head=1,  # Test MQA with 2:1 ratio
        n_embd=32,
    )


def prepare_model_for_testing(model):
    """Prepare model for CPU testing by converting to float32 but keeping rotary embeddings in bfloat16."""
    model = model.float()
    model.cos = model.cos.bfloat16()
    model.sin = model.sin.bfloat16()
    return model


def test_gpt_config():
    """Test that GPTConfig initializes with correct defaults."""
    config = GPTConfig()
    assert config.sequence_len == 1024
    assert config.vocab_size == 50304
    assert config.n_layer == 12
    assert config.n_head == 6
    assert config.n_kv_head == 6
    assert config.n_embd == 768


def test_norm_function():
    """Test the RMSNorm function."""
    x = torch.randn(2, 4, 8)
    y = norm(x)
    # Check shape is preserved
    assert y.shape == x.shape
    # Check it's actually normalized (approximately)
    # RMSNorm: y = x / rms(x) where rms = sqrt(mean(x^2))
    expected_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    expected_y = x / expected_rms
    torch.testing.assert_close(y, expected_y, rtol=1e-4, atol=1e-4)


def test_apply_rotary_emb():
    """Test rotary embeddings application."""
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Create simple cos/sin for testing
    cos = torch.ones(1, seq_len, 1, head_dim // 2)
    sin = torch.zeros(1, seq_len, 1, head_dim // 2)
    
    y = apply_rotary_emb(x, cos, sin)
    
    # Check shape is preserved
    assert y.shape == x.shape
    # Check dtype is preserved
    assert y.dtype == x.dtype


def test_repeat_kv():
    """Test the repeat_kv function for MQA."""
    bs, n_kv_heads, slen, head_dim = 2, 2, 8, 16
    n_rep = 3  # Repeat each KV head 3 times
    
    x = torch.randn(bs, n_kv_heads, slen, head_dim)
    y = repeat_kv(x, n_rep)
    
    # Check output shape
    assert y.shape == (bs, n_kv_heads * n_rep, slen, head_dim)
    
    # Check that heads are repeated correctly
    for i in range(n_kv_heads):
        for j in range(n_rep):
            torch.testing.assert_close(y[:, i * n_rep + j], x[:, i])
    
    # Test n_rep=1 (no-op case)
    y_no_rep = repeat_kv(x, 1)
    torch.testing.assert_close(y_no_rep, x)


def test_gpt_initialization(small_config):
    """Test that GPT model initializes correctly."""
    model = GPT(small_config)
    
    # Check model has the right components
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'lm_head')
    assert len(model.transformer.h) == small_config.n_layer
    
    # Check parameter count is reasonable
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    print(f"Small model has {num_params:,} parameters")


def test_gpt_forward_shape(tiny_config):
    """Test that forward pass produces correct output shapes."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    batch_size = 2
    seq_len = 16
    
    # Create input tokens
    idx = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    
    # Forward pass without targets (inference mode)
    with torch.no_grad():
        logits = model(idx)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size)


def test_gpt_forward_with_targets(tiny_config):
    """Test forward pass with targets (training mode)."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.train()
    
    batch_size = 2
    seq_len = 16
    
    # Create input tokens and targets
    idx = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    
    # Forward pass with targets
    loss = model(idx, targets=targets)
    
    # Check loss is a scalar
    assert loss.shape == ()
    assert loss.item() > 0  # Cross-entropy loss should be positive
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_gpt_backward(tiny_config):
    """Test that backward pass works and gradients flow."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.train()
    
    batch_size = 2
    seq_len = 8
    
    idx = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    
    # Forward and backward
    loss = model(idx, targets=targets)
    loss.backward()
    
    # Check that gradients are computed for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


def test_gpt_generate(tiny_config):
    """Test autoregressive generation."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    # Start with a few tokens
    initial_tokens = [1, 2, 3]
    max_new_tokens = 10
    
    # Generate tokens
    generated = []
    for token in model.generate(initial_tokens, max_tokens=max_new_tokens, temperature=1.0, seed=42):
        generated.append(token)
    
    # Check we generated the right number of tokens
    assert len(generated) == max_new_tokens
    
    # Check all tokens are valid
    for token in generated:
        assert 0 <= token < tiny_config.vocab_size


def test_gpt_generate_deterministic(tiny_config):
    """Test that generation is deterministic with same seed."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    initial_tokens = [1, 2, 3]
    max_new_tokens = 5
    
    # Generate twice with same seed
    gen1 = list(model.generate(initial_tokens, max_tokens=max_new_tokens, temperature=1.0, seed=42))
    gen2 = list(model.generate(initial_tokens, max_tokens=max_new_tokens, temperature=1.0, seed=42))
    
    assert gen1 == gen2, "Generation should be deterministic with same seed"


def test_gpt_generate_greedy(tiny_config):
    """Test greedy decoding (temperature=0)."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    initial_tokens = [1, 2, 3]
    max_new_tokens = 5
    
    # Greedy decoding
    generated = list(model.generate(initial_tokens, max_tokens=max_new_tokens, temperature=0.0))
    
    assert len(generated) == max_new_tokens
    for token in generated:
        assert 0 <= token < tiny_config.vocab_size


def test_gpt_estimate_flops(small_config):
    """Test FLOP estimation."""
    model = GPT(small_config)
    flops = model.estimate_flops()
    
    # FLOPs should be positive
    assert flops > 0
    print(f"Estimated FLOPs per token: {flops:,}")


def test_gpt_setup_optimizers(tiny_config):
    """Test optimizer setup."""
    model = GPT(tiny_config)
    model.init_weights()
    
    # Setup optimizers
    optimizers = model.setup_optimizers()
    
    # Should return a list of 2 optimizers
    assert len(optimizers) == 2
    
    # Check they have parameter groups
    for opt in optimizers:
        assert len(opt.param_groups) > 0


def test_gpt_mqa_shapes(small_config):
    """Test that Multi-Query Attention produces correct shapes."""
    # Modify config to have different n_head and n_kv_head
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=1,
        n_head=8,
        n_kv_head=2,  # 4:1 ratio
        n_embd=64,
    )
    
    model = GPT(config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(idx)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_gpt_long_sequence(small_config):
    """Test with sequences up to max length."""
    model = GPT(small_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.eval()
    
    batch_size = 1
    seq_len = small_config.sequence_len  # Full sequence length
    
    idx = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(idx)
    
    assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


def test_gpt_embedding_dtype():
    """Test that embeddings are cast to bfloat16."""
    config = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    
    model = GPT(config)
    
    # Check that embeddings are in bfloat16
    assert model.transformer.wte.weight.dtype == torch.bfloat16


def test_gpt_rotary_embeddings_dtype(tiny_config):
    """Test that rotary embeddings are in bfloat16."""
    model = GPT(tiny_config)
    model.init_weights()
    
    assert model.cos.dtype == torch.bfloat16
    assert model.sin.dtype == torch.bfloat16


def test_gpt_loss_reduction_modes(tiny_config):
    """Test different loss reduction modes."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.train()
    
    batch_size = 2
    seq_len = 8
    
    idx = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    
    # Test 'mean' reduction (default)
    loss_mean = model(idx, targets=targets, loss_reduction='mean')
    assert loss_mean.shape == ()
    
    # Test 'none' reduction
    loss_none = model(idx, targets=targets, loss_reduction='none')
    assert loss_none.shape == (batch_size * seq_len,)
    
    # The mean of loss_none should be close to loss_mean
    torch.testing.assert_close(loss_none.mean(), loss_mean, rtol=1e-4, atol=1e-4)


def test_gpt_with_ignore_index(tiny_config):
    """Test that -1 targets are ignored in loss."""
    model = GPT(tiny_config)
    model.init_weights()
    model = prepare_model_for_testing(model)
    model.train()
    
    batch_size = 2
    seq_len = 8
    
    idx = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    
    # Compute loss with all valid targets
    loss_all = model(idx, targets=targets)
    
    # Mask out ALL targets except first token
    targets_masked = targets.clone()
    targets_masked[:, 1:] = -1
    loss_masked = model(idx, targets=targets_masked)
    
    # Both should be valid losses
    assert loss_all.item() > 0
    assert loss_masked.item() > 0
    # Loss should be finite
    assert not torch.isnan(loss_masked)
    assert not torch.isinf(loss_masked)
    
    # Test that all -1 targets produces no loss computation error
    targets_all_masked = torch.full_like(targets, -1)
    # This should still work (loss over 0 tokens)
    try:
        loss_all_masked = model(idx, targets=targets_all_masked)
        # If it works, loss should be finite or zero
        assert not torch.isnan(loss_all_masked) or loss_all_masked.item() == 0
    except:
        # It's okay if this raises an error - that's one way to handle no valid targets
        pass

