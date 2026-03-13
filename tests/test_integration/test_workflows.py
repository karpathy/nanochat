"""Integration tests for end-to-end workflows."""

import pytest
import torch
from nanochat.models.gpt import GPT, GPTConfig
from nanochat.data.tokenizer import HuggingFaceTokenizer


def test_tokenizer_model_integration():
    """Test tokenizer encoding works with model forward pass."""
    config = GPTConfig(
        vocab_size=50257,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        sequence_len=128
    )
    model = GPT(config)
    model.eval()
    
    tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode("Hello world")
    input_ids = torch.tensor([tokens[:10]], dtype=torch.long)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    assert logits.shape == (1, input_ids.shape[1], config.vocab_size)


def test_training_checkpoint_integration():
    """Test model save/load preserves weights."""
    import tempfile
    from nanochat.training.checkpoint import save_checkpoint, load_checkpoint
    
    config = GPTConfig(vocab_size=256, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    model = GPT(config)
    initial_weight = model.transformer.h[0].attn.c_q.weight.clone()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            checkpoint_dir=tmpdir,
            step=100,
            model_data=model.state_dict(),
            optimizer_data=None,
            meta_data={"step": 100},
            rank=0
        )
        
        model_data, _, _ = load_checkpoint(tmpdir, step=100, device="cpu")
        model.load_state_dict(model_data)
        
        restored_weight = model.transformer.h[0].attn.c_q.weight
        assert torch.allclose(initial_weight, restored_weight)


def test_kv_cache_integration():
    """Test KVCache works with model forward pass."""
    from nanochat.evaluation.engine import KVCache
    
    config = GPTConfig(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_kv_head=2,
        sequence_len=128
    )
    model = GPT(config)
    model.eval()
    
    # Create KV cache
    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=128,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device="cpu",
        dtype=torch.float32
    )
    
    # First forward pass (prefill)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        logits1 = model(input_ids, kv_cache=kv_cache)
    
    assert logits1.shape == (1, 10, config.vocab_size)
    assert kv_cache.get_pos() == 10
    
    # Second forward pass (decode)
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    with torch.no_grad():
        logits2 = model(next_token, kv_cache=kv_cache)
    
    assert logits2.shape == (1, 1, config.vocab_size)
    assert kv_cache.get_pos() == 11
