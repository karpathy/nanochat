"""
Test UNet Engine functionality.

Run with:
python -m pytest tests/test_unet_engine.py -v
"""

import torch
import pytest
from dataclasses import dataclass


# Mock classes for testing without a real UNet model
@dataclass
class MockUNetConfig:
    """Minimal UNet config for testing."""
    n_layer: tuple = (4, 8)  # Two stages
    n_head: tuple = (4, 8)
    n_kv_head: tuple = (4, 8)
    n_embd: tuple = (64, 128)
    sequence_len: int = 128
    vocab_size: int = 256


class MockUNet:
    """Mock UNet model for testing KV cache without full model."""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.config = MockUNetConfig()
        self._device = torch.device("cpu")
        
        # Mock components that UNet has
        self.encoder = {"transformer_0": [], "transformer_1": []}
        self.decoder = {"transformer_0": [], "transformer_1": []}
        self.n_stage = 2
        
        # Mock embeddings
        self.cos = torch.randn(1, 128, 1, 16)
        self.sin = torch.randn(1, 128, 1, 16)
    
    def get_device(self):
        return self._device
    
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Mock generate that just returns random tokens."""
        for _ in range(max_tokens):
            yield torch.randint(0, self.vocab_size, (1,)).item()


class MockTokenizer:
    """Simple mock tokenizer."""
    
    def __init__(self):
        self._bos = 0
    
    def get_bos_token_id(self):
        return self._bos
    
    def encode(self, text, prepend=None):
        # Simple byte encoding
        tokens = [ord(c) % 256 for c in text]
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens
    
    def decode(self, tokens):
        # Simple decode
        return ''.join([chr(t) if t < 256 else '?' for t in tokens])
    
    def encode_special(self, token):
        # Mock special tokens
        specials = {
            "<|assistant_end|>": 255,
        }
        return specials.get(token, 0)


def test_unet_kv_cache_basic():
    """Test basic UNetKVCache functionality."""
    from inference.unet_engine import UNetKVCache
    
    config = MockUNetConfig()
    batch_size = 2
    max_seq_len = 128
    
    cache = UNetKVCache(
        batch_size=batch_size,
        config=config,
        max_seq_len=max_seq_len,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    # Check initial state
    assert cache.get_pos() == 0
    assert cache.n_stages == 2
    
    # Check cache structure (flat layer caches)
    assert len(cache.layer_caches) == cache.n_layers
    assert len(cache.layer_to_stage) == cache.n_layers
    
    # Test advance
    cache.advance(10)
    assert cache.get_pos() == 10
    
    cache.advance(1)
    assert cache.get_pos() == 11
    
    # Test reset
    cache.reset()
    assert cache.get_pos() == 0


def test_unet_kv_cache_prefill():
    """Test cache prefilling (for multi-sample generation)."""
    from inference.unet_engine import UNetKVCache
    
    config = MockUNetConfig()
    
    # Create source cache
    src_cache = UNetKVCache(
        batch_size=1,
        config=config,
        max_seq_len=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    # Simulate prefill
    src_cache.advance(64)
    
    # Create destination cache for multiple samples
    dst_cache = UNetKVCache(
        batch_size=4,
        config=config,
        max_seq_len=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    # Prefill
    dst_cache.prefill(src_cache)
    
    # Check position was copied
    assert dst_cache.get_pos() == 64


def test_unet_engine_initialization():
    """Test UNetEngine can be initialized."""
    from inference.unet_engine import UNetEngine
    
    model = MockUNet()
    tokenizer = MockTokenizer()
    
    engine = UNetEngine(model, tokenizer)
    
    assert engine.model is model
    assert engine.tokenizer is tokenizer


def test_sample_next_token():
    """Test token sampling functions."""
    from inference.unet_engine import sample_next_token
    
    # Test greedy (temperature=0)
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    rng = torch.Generator()
    tokens = sample_next_token(logits, rng, temperature=0.0)
    assert tokens.item() == 1  # Argmax
    
    # Test top-k
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]])
    tokens = sample_next_token(logits, rng, temperature=1.0, top_k=2)
    assert tokens.item() in [1, 4]  # Top 2 indices


def test_row_state():
    """Test RowState tracking."""
    from inference.unet_engine import RowState
    
    state = RowState([1, 2, 3])
    
    assert state.current_tokens == [1, 2, 3]
    assert len(state.forced_tokens) == 0
    assert not state.in_python_block
    assert not state.completed
    
    # Add forced tokens
    state.forced_tokens.append(4)
    state.forced_tokens.append(5)
    assert len(state.forced_tokens) == 2
    
    # Pop them
    assert state.forced_tokens.popleft() == 4
    assert state.forced_tokens.popleft() == 5


def test_benchmark_result():
    """Test BenchmarkResult dataclass."""
    from inference.benchmark import BenchmarkResult
    
    result = BenchmarkResult(
        prompt_length=64,
        num_generated=32,
        batch_size=1,
        prefill_time=0.1,
        decode_time=0.5,
        total_time=0.6,
        prefill_tokens_per_sec=640.0,
        decode_tokens_per_sec=64.0,
        overall_tokens_per_sec=160.0,
        peak_memory_mb=1024.0,
        memory_before_mb=512.0,
        memory_after_mb=800.0,
    )
    
    # Test string representation
    s = str(result)
    assert "Benchmark Result" in s
    assert "64 tokens" in s
    
    # Test to_dict
    d = result.to_dict()
    assert d['prompt_length'] == 64
    assert d['num_generated'] == 32
    assert d['batch_size'] == 1


def test_comparison_result():
    """Test ComparisonResult dataclass."""
    from inference.benchmark import BenchmarkResult, ComparisonResult
    
    naive = BenchmarkResult(
        prompt_length=64, num_generated=32, batch_size=1,
        prefill_time=0.1, decode_time=1.0, total_time=1.1,
        prefill_tokens_per_sec=640.0, decode_tokens_per_sec=32.0,
        overall_tokens_per_sec=87.3,
        peak_memory_mb=512.0, memory_before_mb=256.0, memory_after_mb=512.0,
    )
    
    engine = BenchmarkResult(
        prompt_length=64, num_generated=32, batch_size=1,
        prefill_time=0.05, decode_time=0.4, total_time=0.45,
        prefill_tokens_per_sec=1280.0, decode_tokens_per_sec=80.0,
        overall_tokens_per_sec=213.3,
        peak_memory_mb=768.0, memory_before_mb=256.0, memory_after_mb=768.0,
    )
    
    comparison = ComparisonResult(
        naive_result=naive,
        engine_result=engine,
        outputs_match=True,
        speedup=1.1 / 0.45,
        memory_overhead_mb=768.0 - 512.0,
    )
    
    assert comparison.outputs_match
    assert comparison.speedup > 2.0
    assert comparison.memory_overhead_mb == 256.0
    
    s = str(comparison)
    assert "Comparison" in s
    assert "Speedup" in s


def test_simple_timer():
    """Test SimpleTimer utility."""
    from inference.profiler import SimpleTimer
    import time
    
    timer = SimpleTimer(device_type="cpu")
    
    # Measure some work
    with timer.measure("test"):
        time.sleep(0.01)
    
    with timer.measure("test"):
        time.sleep(0.01)
    
    # Check measurements
    assert timer.get_count("test") == 2
    mean = timer.get_mean("test")
    assert mean > 5.0  # Should be ~10ms, give some slack
    
    # Check summary
    summary = timer.summary()
    assert "test" in summary
    assert "2" in summary  # count


def test_memory_profiler():
    """Test MemoryProfiler utility."""
    from inference.profiler import MemoryProfiler
    
    # Test CPU mode (snapshots disabled)
    profiler_cpu = MemoryProfiler(device_type="cpu")
    profiler_cpu.snapshot("start")
    profiler_cpu.snapshot("end")
    assert len(profiler_cpu.snapshots) == 0  # No snapshots on CPU
    assert not profiler_cpu.enabled
    
    summary = profiler_cpu.summary()
    assert "not available" in summary
    
    # Test CUDA mode if available
    if torch.cuda.is_available():
        profiler_cuda = MemoryProfiler(device_type="cuda")
        profiler_cuda.snapshot("start")
        profiler_cuda.snapshot("end")
        assert len(profiler_cuda.snapshots) == 2  # Snapshots work on CUDA
        assert profiler_cuda.enabled
        
        summary_cuda = profiler_cuda.summary()
        assert "Memory Profile" in summary_cuda


if __name__ == "__main__":
    # Run tests without pytest
    print("Running UNet Engine tests...")
    
    test_unet_kv_cache_basic()
    print("✓ test_unet_kv_cache_basic")
    
    test_unet_kv_cache_prefill()
    print("✓ test_unet_kv_cache_prefill")
    
    test_unet_engine_initialization()
    print("✓ test_unet_engine_initialization")
    
    test_sample_next_token()
    print("✓ test_sample_next_token")
    
    test_row_state()
    print("✓ test_row_state")
    
    test_benchmark_result()
    print("✓ test_benchmark_result")
    
    test_comparison_result()
    print("✓ test_comparison_result")
    
    test_simple_timer()
    print("✓ test_simple_timer")
    
    test_memory_profiler()
    print("✓ test_memory_profiler")
    
    print("\n✓ All tests passed!")
