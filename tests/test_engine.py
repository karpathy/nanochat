"""
Tests for the inference engine with KV cache and tool use.

Run with:
python -m pytest tests/test_engine.py -v -s --timeout=60
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine, KVCache, use_calculator, sample_next_token
from nanochat.tokenizer import RustBPETokenizer


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
    )
    model = GPT(config)
    model.init_weights()
    # Prepare for CPU testing
    model = model.float()
    model.cos = model.cos.bfloat16()
    model.sin = model.sin.bfloat16()
    model.eval()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def encode(self, text):
            """Simple encode: just return char codes."""
            if isinstance(text, str):
                return [ord(c) % 256 for c in text]
            elif isinstance(text, list):
                return [self.encode(t) for t in text]
        
        def decode(self, ids):
            """Simple decode: convert back to chars."""
            return ''.join(chr(i) for i in ids)
        
        def encode_special(self, token):
            """Encode special tokens to specific IDs."""
            special_map = {
                '<|python_start|>': 250,
                '<|python_end|>': 251,
                '<|output_start|>': 252,
                '<|output_end|>': 253,
                '<|assistant_end|>': 254,
                '<|bos|>': 255,
            }
            return special_map.get(token, 0)
        
        def get_bos_token_id(self):
            return 255
    
    return MockTokenizer()


def test_use_calculator():
    """Test calculator functionality."""
    # Basic arithmetic
    assert use_calculator("2+2") == 4
    assert use_calculator("10-3") == 7
    assert use_calculator("4*5") == 20
    assert use_calculator("15/3") == 5
    
    # With spaces
    assert use_calculator("2 + 2") == 4
    
    # Order of operations
    assert use_calculator("2+3*4") == 14
    
    # Parentheses
    assert use_calculator("(2+3)*4") == 20
    
    # Decimals
    result = use_calculator("10.5+2.5")
    assert result == 13.0
    
    # Commas should be removed
    result = use_calculator("1,000+500")
    assert result == 1500


def test_use_calculator_invalid():
    """Test calculator with invalid inputs."""
    # Non-numeric characters should fail
    assert use_calculator("abc") is None
    assert use_calculator("2+x") is None
    assert use_calculator("import os") is None
    
    # Power operator disabled
    assert use_calculator("2**10") is None
    
    # Division by zero should fail gracefully
    assert use_calculator("1/0") is None


def test_kv_cache_initialization():
    """Test KV cache initialization."""
    batch_size, num_heads, seq_len, head_dim, num_layers = 2, 4, 128, 16, 3
    
    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)
    
    assert cache.pos == 0
    assert cache.kv_cache is None  # Lazy initialization
    assert cache.kv_shape == (num_layers, 2, batch_size, num_heads, seq_len, head_dim)


def test_kv_cache_insert_and_retrieve():
    """Test inserting and retrieving from KV cache."""
    batch_size, num_heads, seq_len, head_dim, num_layers = 1, 2, 32, 8, 2
    
    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)
    
    # Create some keys and values
    k = torch.randn(batch_size, num_heads, 4, head_dim)
    v = torch.randn(batch_size, num_heads, 4, head_dim)
    
    # Insert for layer 0
    k_out, v_out = cache.insert_kv(0, k, v)
    
    # Should return views of size 4 (what we inserted)
    assert k_out.shape == (batch_size, num_heads, 4, head_dim)
    assert v_out.shape == (batch_size, num_heads, 4, head_dim)
    
    # Values should match
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)
    
    # Position should not advance until last layer
    assert cache.pos == 0
    
    # Insert for layer 1 (last layer)
    k_out, v_out = cache.insert_kv(1, k, v)
    
    # Now position should advance
    assert cache.pos == 4


def test_kv_cache_sequential_inserts():
    """Test sequential token generation with KV cache."""
    batch_size, num_heads, seq_len, head_dim, num_layers = 1, 2, 64, 8, 1
    
    cache = KVCache(batch_size, num_heads, seq_len, head_dim, num_layers)
    
    # Insert tokens one at a time
    for i in range(5):
        k = torch.randn(batch_size, num_heads, 1, head_dim)
        v = torch.randn(batch_size, num_heads, 1, head_dim)
        
        k_out, v_out = cache.insert_kv(0, k, v)
        
        # Should return all keys/values so far
        assert k_out.shape == (batch_size, num_heads, i + 1, head_dim)
        assert v_out.shape == (batch_size, num_heads, i + 1, head_dim)
    
    assert cache.pos == 5


def test_kv_cache_reset():
    """Test resetting KV cache."""
    cache = KVCache(1, 2, 32, 8, 1)
    
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    cache.insert_kv(0, k, v)
    
    assert cache.pos == 4
    
    cache.reset()
    assert cache.pos == 0


def test_kv_cache_prefill():
    """Test prefilling KV cache from another cache."""
    # Create a small cache sized exactly for the data we'll insert
    # (This matches the actual usage pattern in engine.py)
    num_tokens = 4
    small_cache = KVCache(1, 2, num_tokens, 8, 2)
    k = torch.randn(1, 2, num_tokens, 8)
    v = torch.randn(1, 2, num_tokens, 8)
    small_cache.insert_kv(0, k, v)
    small_cache.insert_kv(1, k, v)
    
    assert small_cache.pos == num_tokens
    
    # Create a larger cache and prefill from small
    large_cache = KVCache(1, 2, 128, 8, 2)
    large_cache.prefill(small_cache)
    
    assert large_cache.pos == small_cache.pos
    assert large_cache.pos == num_tokens


def test_kv_cache_dynamic_growth():
    """Test that KV cache grows dynamically."""
    cache = KVCache(1, 2, 16, 8, 1)  # Start with small size
    
    # Insert more tokens than initial capacity
    for i in range(20):
        k = torch.randn(1, 2, 1, 8)
        v = torch.randn(1, 2, 1, 8)
        k_out, v_out = cache.insert_kv(0, k, v)
        
        assert k_out.shape[2] == i + 1  # Should have all tokens so far
    
    # Cache should have grown
    assert cache.kv_cache.shape[4] >= 20


def test_sample_next_token_greedy():
    """Test greedy sampling (temperature=0)."""
    vocab_size = 10
    batch_size = 2
    
    logits = torch.randn(batch_size, vocab_size)
    rng = torch.Generator()
    rng.manual_seed(42)
    
    tokens = sample_next_token(logits, rng, temperature=0.0)
    
    assert tokens.shape == (batch_size, 1)
    
    # Should be argmax
    expected = torch.argmax(logits, dim=-1, keepdim=True)
    torch.testing.assert_close(tokens, expected)


def test_sample_next_token_with_temperature():
    """Test sampling with temperature."""
    vocab_size = 10
    batch_size = 2
    
    logits = torch.randn(batch_size, vocab_size)
    rng = torch.Generator()
    rng.manual_seed(42)
    
    tokens = sample_next_token(logits, rng, temperature=1.0)
    
    assert tokens.shape == (batch_size, 1)
    assert torch.all((tokens >= 0) & (tokens < vocab_size))


def test_sample_next_token_top_k():
    """Test top-k sampling."""
    vocab_size = 100
    batch_size = 1
    
    logits = torch.randn(batch_size, vocab_size)
    rng = torch.Generator()
    rng.manual_seed(42)
    
    # Sample multiple times and check all are in top-k
    top_k = 5
    samples = []
    for _ in range(20):
        rng.manual_seed(42 + _)
        token = sample_next_token(logits, rng, temperature=1.0, top_k=top_k)
        samples.append(token.item())
    
    # Get the actual top-k indices
    _, top_k_indices = torch.topk(logits[0], top_k)
    top_k_set = set(top_k_indices.tolist())
    
    # All samples should be in top-k
    for sample in samples:
        assert sample in top_k_set


def test_engine_initialization(tiny_model, mock_tokenizer):
    """Test Engine initialization."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    assert engine.model is tiny_model
    assert engine.tokenizer is mock_tokenizer


def test_engine_generate_batch(tiny_model, mock_tokenizer):
    """Test batch generation with Engine."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    # Start with some tokens
    initial_tokens = [1, 2, 3, 4]
    num_samples = 2
    max_tokens = 10
    
    results, masks = engine.generate_batch(
        initial_tokens,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=1.0,
        seed=42
    )
    
    # Should return num_samples results
    assert len(results) == num_samples
    assert len(masks) == num_samples
    
    # Each result should have at least the initial tokens
    for result, mask in zip(results, masks):
        assert len(result) >= len(initial_tokens)
        assert len(result) == len(mask)
        # Initial tokens should match
        assert result[:len(initial_tokens)] == initial_tokens


def test_engine_generate_deterministic(tiny_model, mock_tokenizer):
    """Test that generation is deterministic with same seed."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3]
    
    results1, _ = engine.generate_batch(initial_tokens, num_samples=1, max_tokens=5, seed=42)
    results2, _ = engine.generate_batch(initial_tokens, num_samples=1, max_tokens=5, seed=42)
    
    assert results1[0] == results2[0], "Results should be identical with same seed"


def test_engine_generate_streaming(tiny_model, mock_tokenizer):
    """Test streaming generation."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3]
    max_tokens = 5
    
    token_columns = []
    mask_columns = []
    
    for token_col, mask_col in engine.generate(
        initial_tokens, 
        num_samples=2, 
        max_tokens=max_tokens,
        seed=42
    ):
        token_columns.append(token_col)
        mask_columns.append(mask_col)
    
    # Should generate max_tokens columns
    assert len(token_columns) == max_tokens
    
    # Each column should have 2 tokens (num_samples=2)
    for col in token_columns:
        assert len(col) == 2


def test_engine_greedy_decode(tiny_model, mock_tokenizer):
    """Test greedy decoding."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3]
    
    results, _ = engine.generate_batch(
        initial_tokens, 
        num_samples=1,
        max_tokens=5,
        temperature=0.0  # Greedy
    )
    
    assert len(results) == 1
    assert len(results[0]) >= len(initial_tokens)


def test_engine_max_tokens_limit(tiny_model, mock_tokenizer):
    """Test that generation respects max_tokens."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3]
    max_tokens = 3
    
    results, _ = engine.generate_batch(
        initial_tokens,
        num_samples=1,
        max_tokens=max_tokens,
        seed=42
    )
    
    # Should not exceed initial + max_tokens
    assert len(results[0]) <= len(initial_tokens) + max_tokens


def test_engine_multiple_samples(tiny_model, mock_tokenizer):
    """Test generating multiple samples in parallel."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3]
    num_samples = 4
    
    results, _ = engine.generate_batch(
        initial_tokens,
        num_samples=num_samples,
        max_tokens=5,
        temperature=1.0,  # Non-zero temp for diversity
        seed=42
    )
    
    assert len(results) == num_samples
    
    # All should start with same initial tokens
    for result in results:
        assert result[:len(initial_tokens)] == initial_tokens


def test_engine_with_kv_cache(tiny_model, mock_tokenizer):
    """Test that KV cache is used during generation."""
    engine = Engine(tiny_model, mock_tokenizer)
    
    initial_tokens = [1, 2, 3, 4, 5]
    
    # Generate with a longer prompt to benefit from KV cache
    results, _ = engine.generate_batch(
        initial_tokens,
        num_samples=1,
        max_tokens=5,
        seed=42
    )
    
    # Should successfully generate
    assert len(results) == 1
    assert len(results[0]) > len(initial_tokens)

