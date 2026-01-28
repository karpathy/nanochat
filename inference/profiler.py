"""
Profiling utilities for detailed performance analysis.

Provides tools for:
- PyTorch profiler integration
- CUDA kernel profiling
- Memory profiling
- Layer-wise timing breakdown
"""

import torch
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


def get_checkpoint_profile_dir(model_tag: Optional[str] = None) -> str:
    """
    Get the directory for saving profiling traces, based on checkpoint location.
    
    Args:
        model_tag: Model tag (e.g., 'd[6,12]-h[768,1536]')
    
    Returns:
        Path to profile directory
    """
    try:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()
    except:
        base_dir = os.getcwd()
    
    if model_tag:
        profile_dir = os.path.join(base_dir, "base_checkpoints", model_tag, "profiles")
    else:
        profile_dir = os.path.join(base_dir, "profiles")
    
    os.makedirs(profile_dir, exist_ok=True)
    return profile_dir


def generate_profile_filename(
    prefix: str = "trace",
    prompt_length: Optional[int] = None,
    max_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    model_tag: Optional[str] = None,
    extension: str = "json",
) -> Tuple[str, str]:
    """
    Generate an informative filename for profiling traces.
    
    Args:
        prefix: Prefix for filename (e.g., 'trace', 'profile')
        prompt_length: Length of input prompt
        max_tokens: Number of generated tokens
        batch_size: Batch size used
        temperature: Sampling temperature
        model_tag: Model identifier
        extension: File extension (default: 'json')
    
    Returns:
        Tuple of (directory, filename)
    """
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename parts
    parts = [prefix, timestamp]
    
    if prompt_length is not None:
        parts.append(f"prompt{prompt_length}")
    
    if max_tokens is not None:
        parts.append(f"gen{max_tokens}")
    
    if batch_size is not None and batch_size > 1:
        parts.append(f"bs{batch_size}")
    
    if temperature is not None:
        temp_str = f"t{temperature:.1f}".replace(".", "p")
        parts.append(temp_str)
    
    filename = "_".join(parts) + f".{extension}"
    
    # Get directory
    directory = get_checkpoint_profile_dir(model_tag)
    
    return directory, filename


@dataclass
class ProfileResult:
    """Results from profiling a model."""
    total_time_ms: float
    gpu_time_ms: float
    cpu_time_ms: float
    memory_mb: float
    
    # Detailed breakdowns (if available)
    layer_times: Optional[Dict[str, float]] = None
    kernel_times: Optional[Dict[str, float]] = None
    
    # Trace file path (if saved)
    trace_path: Optional[str] = None
    
    def __str__(self):
        lines = [
            "=" * 70,
            "Profile Result",
            "=" * 70,
            f"Total time: {self.total_time_ms:.2f} ms",
            f"GPU time:   {self.gpu_time_ms:.2f} ms",
            f"CPU time:   {self.cpu_time_ms:.2f} ms",
            f"Memory:     {self.memory_mb:.1f} MB",
        ]
        
        if self.trace_path:
            lines.append(f"Trace saved: {self.trace_path}")
        
        if self.layer_times:
            lines.append("\nTop 10 Layers by Time:")
            sorted_layers = sorted(self.layer_times.items(), key=lambda x: x[1], reverse=True)[:10]
            for name, time_ms in sorted_layers:
                lines.append(f"  {name[:50]:<50} {time_ms:>8.2f} ms")
        
        if self.kernel_times:
            lines.append("\nTop 10 CUDA Kernels by Time:")
            sorted_kernels = sorted(self.kernel_times.items(), key=lambda x: x[1], reverse=True)[:10]
            for name, time_ms in sorted_kernels:
                lines.append(f"  {name[:50]:<50} {time_ms:>8.2f} ms")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class TorchProfiler:
    """
    Wrapper around torch.profiler for easy profiling of model operations.
    """
    
    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        self.enabled = device_type == "cuda"
    
    @contextmanager
    def profile(
        self,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = True,
        output_path: Optional[str] = None,
    ):
        """
        Context manager for profiling a code block.
        
        Args:
            with_stack: Include Python stack traces
            with_flops: Count FLOPs (experimental)
            with_modules: Record module hierarchy
            output_path: If provided, save chrome trace to this path
        
        Yields:
            Profiler object (can be used to export results)
        
        Example:
            with profiler.profile(output_path="trace.json") as prof:
                model(inputs)
            # trace.json is now saved
        """
        if not self.enabled:
            yield None
            return
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device_type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        ) as prof:
            yield prof
        
        if output_path:
            prof.export_chrome_trace(output_path)
            print(f"Saved profiler trace to: {output_path}")
    
    def profile_function(
        self,
        func: Callable,
        *args,
        warmup_iters: int = 3,
        profile_iters: int = 10,
        output_path: Optional[str] = None,
        auto_save: bool = True,
        prompt_length: Optional[int] = None,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        temperature: Optional[float] = None,
        model_tag: Optional[str] = None,
        **kwargs,
    ) -> ProfileResult:
        """
        Profile a function over multiple iterations.
        
        Args:
            func: Function to profile
            *args: Positional arguments to func
            warmup_iters: Number of warmup iterations (not profiled)
            profile_iters: Number of iterations to profile
            output_path: Optional explicit path to save chrome trace (overrides auto_save)
            auto_save: If True and output_path is None, auto-generate filename in checkpoint dir
            prompt_length: Prompt length (for filename generation)
            max_tokens: Max tokens to generate (for filename generation)
            batch_size: Batch size (for filename generation)
            temperature: Sampling temperature (for filename generation)
            model_tag: Model tag (for directory selection)
            **kwargs: Keyword arguments to func
        
        Returns:
            ProfileResult with timing and memory statistics
        """
        # Auto-generate trace path if needed
        trace_path = output_path
        if trace_path is None and auto_save and self.enabled:
            directory, filename = generate_profile_filename(
                prefix="profile",
                prompt_length=prompt_length,
                max_tokens=max_tokens,
                batch_size=batch_size,
                temperature=temperature,
                model_tag=model_tag,
            )
            trace_path = os.path.join(directory, filename)
        
        if not self.enabled:
            # Simple timing fallback for non-CUDA
            for _ in range(warmup_iters):
                func(*args, **kwargs)
            
            start = time.time()
            for _ in range(profile_iters):
                func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000 / profile_iters  # ms per iteration
            
            return ProfileResult(
                total_time_ms=elapsed,
                gpu_time_ms=0.0,
                cpu_time_ms=elapsed,
                memory_mb=0.0,
                trace_path=None,
            )
        
        # Warmup
        for _ in range(warmup_iters):
            func(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        # Profile
        with self.profile(output_path=trace_path) as prof:
            for _ in range(profile_iters):
                func(*args, **kwargs)
                torch.cuda.synchronize()
        
        # Extract statistics
        events = prof.key_averages()
        
        total_gpu_time = 0.0
        total_cpu_time = 0.0
        layer_times = {}
        kernel_times = {}
        
        for event in events:
            if event.device_type == torch.profiler.DeviceType.CUDA:
                total_gpu_time += event.cuda_time_total / profile_iters / 1000  # convert to ms
                if event.key:
                    kernel_times[event.key] = event.cuda_time_total / profile_iters / 1000
            
            total_cpu_time += event.cpu_time_total / profile_iters / 1000
            
            # Track module times if available
            if hasattr(event, 'module_hierarchy') and event.module_hierarchy:
                layer_name = event.module_hierarchy
                if layer_name not in layer_times:
                    layer_times[layer_name] = 0.0
                layer_times[layer_name] += event.cuda_time_total / profile_iters / 1000
        
        # Get memory usage
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if self.enabled else 0.0
        
        return ProfileResult(
            total_time_ms=(total_gpu_time + total_cpu_time) / 2,  # Average
            gpu_time_ms=total_gpu_time,
            cpu_time_ms=total_cpu_time,
            memory_mb=memory_mb,
            layer_times=layer_times if layer_times else None,
            kernel_times=kernel_times if kernel_times else None,
            trace_path=trace_path if trace_path else None,
        )
    
    def print_summary(self, prof, sort_by: str = "cuda_time_total", row_limit: int = 20):
        """
        Print a formatted summary of profiling results.
        
        Args:
            prof: Profiler object from profile() context manager
            sort_by: Metric to sort by (cuda_time_total, cpu_time_total, etc.)
            row_limit: Maximum number of rows to display
        """
        if prof is None:
            print("Profiling not available on this device")
            return
        
        print("\n" + "=" * 100)
        print("Profiler Summary")
        print("=" * 100)
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


class SimpleTimer:
    """
    Simple timer for measuring wall-clock time of code blocks.
    Useful for quick timing without full profiling overhead.
    """
    
    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        if device_type == "cuda":
            self.synchronize = torch.cuda.synchronize
        else:
            self.synchronize = lambda: None
        
        self.times: Dict[str, List[float]] = {}
    
    @contextmanager
    def measure(self, name: str):
        """
        Context manager to time a block of code.
        
        Args:
            name: Label for this timing measurement
        
        Example:
            timer = SimpleTimer()
            with timer.measure("forward"):
                model(inputs)
            print(f"Forward took {timer.get_mean('forward'):.2f}ms")
        """
        self.synchronize()
        start = time.time()
        yield
        self.synchronize()
        elapsed_ms = (time.time() - start) * 1000
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed_ms)
    
    def get_mean(self, name: str) -> float:
        """Get mean time for a measurement."""
        if name not in self.times or not self.times[name]:
            return 0.0
        return sum(self.times[name]) / len(self.times[name])
    
    def get_total(self, name: str) -> float:
        """Get total time for a measurement."""
        if name not in self.times:
            return 0.0
        return sum(self.times[name])
    
    def get_count(self, name: str) -> int:
        """Get number of measurements for a name."""
        if name not in self.times:
            return 0
        return len(self.times[name])
    
    def reset(self):
        """Clear all measurements."""
        self.times.clear()
    
    def summary(self) -> str:
        """Get a formatted summary of all measurements."""
        if not self.times:
            return "No measurements recorded"
        
        lines = [
            "=" * 70,
            "Timer Summary",
            "=" * 70,
            f"{'Name':<30} {'Count':>8} {'Mean':>12} {'Total':>12}",
            "-" * 70,
        ]
        
        for name in sorted(self.times.keys()):
            count = self.get_count(name)
            mean = self.get_mean(name)
            total = self.get_total(name)
            lines.append(f"{name:<30} {count:>8} {mean:>10.2f}ms {total:>10.2f}ms")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class MemoryProfiler:
    """
    Track memory usage throughout model execution.
    """
    
    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type
        self.enabled = device_type == "cuda"
        self.snapshots: List[Tuple[str, float]] = []
    
    def reset(self):
        """Reset memory tracking."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
        self.snapshots.clear()
    
    def snapshot(self, label: str):
        """Take a memory snapshot with a label."""
        if not self.enabled:
            return
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        self.snapshots.append((label, allocated))
    
    def get_peak(self) -> float:
        """Get peak memory usage in MB."""
        if not self.enabled:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    def get_current(self) -> float:
        """Get current memory usage in MB."""
        if not self.enabled:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)
    
    def summary(self) -> str:
        """Get formatted summary of memory snapshots."""
        if not self.enabled:
            return "Memory profiling not available on this device"
        
        lines = [
            "=" * 70,
            "Memory Profile",
            "=" * 70,
            f"Current: {self.get_current():.1f} MB",
            f"Peak:    {self.get_peak():.1f} MB",
        ]
        
        if self.snapshots:
            lines.append("\nSnapshots:")
            for label, mem_mb in self.snapshots:
                lines.append(f"  {label:<40} {mem_mb:>10.1f} MB")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def profile_model_inference(
    model,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 32,
    batch_size: int = 1,
    temperature: float = 0.0,
    warmup_iters: int = 3,
    profile_iters: int = 10,
    output_path: Optional[str] = None,
    auto_save: bool = True,
    model_tag: Optional[str] = None,
    device_type: str = "cuda",
    use_engine: bool = False,
    engine = None,
) -> ProfileResult:
    """
    Convenience function to profile model inference.
    
    Args:
        model: Model to profile
        tokenizer: Tokenizer (for decoding)
        prompt_tokens: Input prompt as list of token IDs
        max_tokens: Number of tokens to generate
        batch_size: Batch size for generation
        temperature: Sampling temperature
        warmup_iters: Warmup iterations
        profile_iters: Profiling iterations
        output_path: Optional explicit path to save chrome trace (overrides auto_save)
        auto_save: If True and output_path is None, auto-generate filename
        model_tag: Model identifier (for auto-generated filename)
        device_type: Device type (cuda/cpu/mps)
        use_engine: If True, profile engine.generate() instead of model.generate()
        engine: Engine instance (required if use_engine=True)
    
    Returns:
        ProfileResult with timing and memory info
    """
    profiler = TorchProfiler(device_type=device_type)
    
    # Adjust filename prefix based on method
    filename_prefix = "profile_engine" if use_engine else "profile_naive"
    
    def run_inference():
        with torch.no_grad():
            if use_engine:
                if engine is None:
                    raise ValueError("engine parameter required when use_engine=True")
                # Profile engine generation (with KV cache)
                for token_column, _ in engine.generate(
                    prompt_tokens,
                    num_samples=batch_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    pass
            else:
                # Profile naive generation (no KV cache)
                for _ in model.generate(prompt_tokens, max_tokens=max_tokens, temperature=temperature):
                    pass
    
    # Auto-generate filename with appropriate prefix if needed
    trace_path = output_path
    if trace_path is None and auto_save and device_type == "cuda":
        directory, filename = generate_profile_filename(
            prefix=filename_prefix,
            prompt_length=len(prompt_tokens),
            max_tokens=max_tokens,
            batch_size=batch_size,
            temperature=temperature,
            model_tag=model_tag,
        )
        trace_path = os.path.join(directory, filename)
    
    return profiler.profile_function(
        run_inference,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters,
        output_path=trace_path,
        auto_save=False,  # We handle auto_save manually above
        prompt_length=len(prompt_tokens),
        max_tokens=max_tokens,
        batch_size=batch_size,
        temperature=temperature,
        model_tag=model_tag,
    )
