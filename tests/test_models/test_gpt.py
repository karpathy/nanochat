"""Test GPT model."""
import torch
from nanochat.models import GPT, GPTConfig


def test_forward_pass_shapes():
    """Verify output shapes for all depth settings."""
    for depth in [12, 16, 20, 24]:
        config = GPTConfig(
            sequence_len=128,
            vocab_size=256,
            n_layer=depth,
            n_embd=depth * 64,
            n_head=depth,
            n_kv_head=depth,
        )
        model = GPT(config)
        ids = torch.randint(0, 256, (2, 128))
        logits = model(ids)
        assert logits.shape == (2, 128, 256), f"Expected (2, 128, 256), got {logits.shape}"


def test_parameter_count_scaling():
    """Verify parameter count follows expected scaling."""
    config = GPTConfig(
        sequence_len=1024,
        vocab_size=32768,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,
    )
    model = GPT(config)
    params = sum(p.numel() for p in model.parameters())
    # d12 with vocab 32768 and embd 768 is ~286M params (includes embeddings)
    assert 280_000_000 < params < 290_000_000, f"Expected ~286M params, got {params:,}"


def test_model_initialization():
    """Test model can be initialized and moved to device."""
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
    
    # Check weights are initialized (not NaN)
    for param in model.parameters():
        assert not torch.isnan(param).any(), "Model has NaN weights after initialization"


def test_loss_computation():
    """Test loss computation with targets."""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=2,
        n_embd=128,
        n_head=2,
        n_kv_head=2,
    )
    model = GPT(config)
    model.init_weights()
    
    inputs = torch.randint(0, 100, (2, 64))
    targets = torch.randint(0, 100, (2, 64))
    
    loss = model(inputs, targets)
    assert loss.ndim == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
