"""
Benchmarking utilities for inference performance measurement.

Provides tools to measure:
- Prefill latency (time to process initial prompt)
- Decode throughput (tokens/sec during generation)
- Memory usage (peak, KV cache size)
- Batch size scaling
- Sequence length scaling
"""

import time
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    # Basic info
    prompt_length: int
    num_generated: int
    batch_size: int
    
    # Timing
    prefill_time: float  # seconds
    decode_time: float   # seconds
    total_time: float    # seconds
    
    # Throughput
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    overall_tokens_per_sec: float
    
    # Memory (in MB)
    peak_memory_mb: float
    memory_before_mb: float
    memory_after_mb: float
    
    # Optional metadata
    temperature: float = 1.0
    top_k: Optional[int] = None
    model_tag: Optional[str] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    def __str__(self):
        lines = [
            "=" * 70,
            "Benchmark Result",
            "=" * 70,
            f"Model: {self.model_tag or 'N/A'}",
            f"Device: {self.device} | Dtype: {self.dtype}",
            f"Prompt length: {self.prompt_length} tokens",
            f"Generated: {self.num_generated} tokens",
            f"Batch size: {self.batch_size}",
            "",
            "Timing:",
            f"  Prefill:  {self.prefill_time:.3f}s ({self.prefill_tokens_per_sec:.1f} tok/s)",
            f"  Decode:   {self.decode_time:.3f}s ({self.decode_tokens_per_sec:.1f} tok/s)",
            f"  Total:    {self.total_time:.3f}s ({self.overall_tokens_per_sec:.1f} tok/s)",
            "",
            "Memory:",
            f"  Before:   {self.memory_before_mb:.1f} MB",
            f"  Peak:     {self.peak_memory_mb:.1f} MB",
            f"  After:    {self.memory_after_mb:.1f} MB",
            f"  Increase: {self.peak_memory_mb - self.memory_before_mb:.1f} MB",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def to_dict(self):
        """Convert to dictionary for easy serialization."""
        return {
            'prompt_length': self.prompt_length,
            'num_generated': self.num_generated,
            'batch_size': self.batch_size,
            'prefill_time': self.prefill_time,
            'decode_time': self.decode_time,
            'total_time': self.total_time,
            'prefill_tokens_per_sec': self.prefill_tokens_per_sec,
            'decode_tokens_per_sec': self.decode_tokens_per_sec,
            'overall_tokens_per_sec': self.overall_tokens_per_sec,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_before_mb': self.memory_before_mb,
            'memory_after_mb': self.memory_after_mb,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'model_tag': self.model_tag,
            'device': self.device,
            'dtype': self.dtype,
        }


@dataclass
class ComparisonResult:
    """Results comparing two generation methods."""
    naive_result: BenchmarkResult
    engine_result: BenchmarkResult
    outputs_match: bool
    speedup: float
    memory_overhead_mb: float
    
    def __str__(self):
        lines = [
            "=" * 70,
            "Comparison: Naive vs Engine",
            "=" * 70,
            f"Outputs match: {'✓ YES' if self.outputs_match else '✗ NO'}",
            f"Speedup: {self.speedup:.2f}x",
            "",
            "Naive (no cache):",
            f"  Total time: {self.naive_result.total_time:.3f}s",
            f"  Throughput: {self.naive_result.overall_tokens_per_sec:.1f} tok/s",
            f"  Peak memory: {self.naive_result.peak_memory_mb:.1f} MB",
            "",
            "Engine (with cache):",
            f"  Total time: {self.engine_result.total_time:.3f}s",
            f"  Throughput: {self.engine_result.overall_tokens_per_sec:.1f} tok/s",
            f"  Peak memory: {self.engine_result.peak_memory_mb:.1f} MB",
            "",
            f"Memory overhead from caching: {self.memory_overhead_mb:.1f} MB",
            "=" * 70,
        ]
        return "\n".join(lines)


class InferenceBenchmark:
    """
    Benchmark runner for inference performance testing.
    """
    
    def __init__(self, model, tokenizer, device, autocast_ctx=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx or (lambda: torch.no_grad())
        self.device_type = device.type
        
        # Setup synchronization and memory tracking
        if self.device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
            self.get_memory = lambda: torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            self.reset_memory = torch.cuda.reset_peak_memory_stats
        else:
            self.synchronize = lambda: None
            self.get_memory = lambda: 0.0
            self.reset_memory = lambda: None
    
    @torch.inference_mode()
    def benchmark_generation(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        use_engine: bool = True,
        engine = None,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        model_tag: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a single generation run.
        
        Args:
            prompt_tokens: Input prompt as list of token IDs
            max_tokens: Number of tokens to generate
            use_engine: Whether to use Engine (True) or naive generate (False)
            engine: Pre-initialized engine object (if use_engine=True)
            batch_size: Batch size for generation
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            model_tag: Model identifier for results
        
        Returns:
            BenchmarkResult with detailed metrics
        """
        self.reset_memory()
        memory_before = self.get_memory()
        
        # Track prefill time separately (only for engine, naive doesn't separate)
        prefill_time = 0.0
        decode_time = 0.0
        generated_tokens = []
        
        self.synchronize()
        total_start = time.time()
        
        with self.autocast_ctx:
            if use_engine and engine is not None:
                # Engine with KV cache: measure prefill + decode separately
                # We can't easily separate them without modifying engine, so we approximate
                # by timing first token (which includes prefill) vs subsequent tokens
                first_token_time = None
                
                for i, (token_column, _) in enumerate(engine.generate(
                    prompt_tokens,
                    num_samples=batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )):
                    if i == 0:
                        self.synchronize()
                        first_token_time = time.time() - total_start
                    generated_tokens.append(token_column[0] if batch_size == 1 else token_column)
                
                self.synchronize()
                total_time = time.time() - total_start
                
                # Approximate: first token includes prefill, rest is pure decode
                if first_token_time is not None:
                    prefill_time = first_token_time
                    decode_time = total_time - first_token_time
                else:
                    prefill_time = 0.0
                    decode_time = total_time
                    
            else:
                # Naive generate (no separation possible)
                ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
                for token in self.model.generate(
                    prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                ):
                    generated_tokens.append(token)
                
                self.synchronize()
                total_time = time.time() - total_start
                
                # Can't separate prefill/decode for naive
                prefill_time = 0.0
                decode_time = total_time
        
        memory_after = self.get_memory()
        peak_memory = self.get_memory()
        
        # Calculate throughput
        num_generated = len(generated_tokens)
        prompt_length = len(prompt_tokens)
        
        if prefill_time > 0:
            prefill_tokens_per_sec = prompt_length / prefill_time
        else:
            prefill_tokens_per_sec = 0.0
        
        if decode_time > 0:
            decode_tokens_per_sec = num_generated / decode_time
        else:
            decode_tokens_per_sec = 0.0
        
        if total_time > 0:
            overall_tokens_per_sec = (prompt_length + num_generated) / total_time
        else:
            overall_tokens_per_sec = 0.0
        
        return BenchmarkResult(
            prompt_length=prompt_length,
            num_generated=num_generated,
            batch_size=batch_size,
            prefill_time=prefill_time,
            decode_time=decode_time,
            total_time=total_time,
            prefill_tokens_per_sec=prefill_tokens_per_sec,
            decode_tokens_per_sec=decode_tokens_per_sec,
            overall_tokens_per_sec=overall_tokens_per_sec,
            peak_memory_mb=peak_memory,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            temperature=temperature,
            top_k=top_k,
            model_tag=model_tag,
            device=self.device_type,
            dtype="bfloat16" if self.device_type == "cuda" else "float32",
        )
    
    def compare_methods(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        engine,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        model_tag: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare naive generation vs engine generation.
        
        Returns comparison results including speedup and correctness check.
        """
        # Run naive version
        print("Running naive generation (no cache)...")
        naive_result = self.benchmark_generation(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            use_engine=False,
            engine=None,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            model_tag=model_tag,
        )
        
        # Collect naive output for comparison
        naive_tokens = []
        with self.autocast_ctx:
            for token in self.model.generate(
                prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                naive_tokens.append(token)
        
        # Run engine version
        print("Running engine generation (with cache)...")
        engine_result = self.benchmark_generation(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            use_engine=True,
            engine=engine,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            model_tag=model_tag,
        )
        
        # Collect engine output for comparison
        engine_tokens = []
        with self.autocast_ctx:
            for token_column, _ in engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                engine_tokens.append(token_column[0])
        
        # Compare outputs
        outputs_match = naive_tokens == engine_tokens
        
        # Calculate metrics
        speedup = naive_result.total_time / engine_result.total_time if engine_result.total_time > 0 else 0.0
        memory_overhead = engine_result.peak_memory_mb - naive_result.peak_memory_mb
        
        return ComparisonResult(
            naive_result=naive_result,
            engine_result=engine_result,
            outputs_match=outputs_match,
            speedup=speedup,
            memory_overhead_mb=memory_overhead,
        )
    
    def sweep_sequence_lengths(
        self,
        engine,
        base_prompt: str,
        lengths: List[int],
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> List[BenchmarkResult]:
        """
        Benchmark across different prompt lengths.
        
        Args:
            engine: Engine instance to benchmark
            base_prompt: Base text prompt (will be repeated to reach target lengths)
            lengths: List of target prompt lengths to test
            max_tokens: Tokens to generate for each test
            temperature: Sampling temperature
        
        Returns:
            List of BenchmarkResult, one per length
        """
        results = []
        
        for target_length in lengths:
            # Create prompt of target length by repeating base prompt
            prompt_tokens = self.tokenizer.encode(base_prompt, prepend=self.tokenizer.get_bos_token_id())
            while len(prompt_tokens) < target_length:
                prompt_tokens.extend(self.tokenizer.encode(base_prompt))
            prompt_tokens = prompt_tokens[:target_length]
            
            print(f"\nBenchmarking with prompt length {len(prompt_tokens)}...")
            result = self.benchmark_generation(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                use_engine=True,
                engine=engine,
                batch_size=1,
                temperature=temperature,
            )
            results.append(result)
            print(f"  Throughput: {result.decode_tokens_per_sec:.1f} tok/s")
        
        return results
    
    def sweep_batch_sizes(
        self,
        engine,
        prompt_tokens: List[int],
        batch_sizes: List[int],
        max_tokens: int = 32,
        temperature: float = 1.0,
    ) -> List[BenchmarkResult]:
        """
        Benchmark across different batch sizes.
        
        Args:
            engine: Engine instance to benchmark
            prompt_tokens: Input prompt
            batch_sizes: List of batch sizes to test
            max_tokens: Tokens to generate for each sample
            temperature: Sampling temperature
        
        Returns:
            List of BenchmarkResult, one per batch size
        """
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking with batch size {batch_size}...")
            result = self.benchmark_generation(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                use_engine=True,
                engine=engine,
                batch_size=batch_size,
                temperature=temperature,
            )
            results.append(result)
            print(f"  Throughput: {result.decode_tokens_per_sec:.1f} tok/s")
            print(f"  Memory: {result.peak_memory_mb:.1f} MB")
        
        return results
