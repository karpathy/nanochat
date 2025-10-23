"""
Benchmarker for comparing tok.c tokenize_fast() vs rust split_text()
Measures speed WITHOUT subprocess overhead - direct function calls only.

Usage:
    cd pytok
    source ../.venv/bin/activate
    python3 bench.py                          # Run synthetic data benchmarks
    python3 bench.py /path/to/file.txt        # Benchmark a specific file
    python3 bench.py file1.txt file2.txt ...  # Benchmark multiple files
"""

import sys
import ctypes
import random
import time
import statistics
from pathlib import Path

from nanochat.tokenizer import SPLIT_PATTERN
from rustbpe import split_text as rust_split_text
from fregex.fuzz import gen_valid_unicode_string, compare_pair_text
from fregex.cload import *

def bench_c_regex(data: bytes, iterations: int) -> list:
    times = []
    for _ in range(iterations):
        token_list = TokenList()
        c_lib.tokenlist_init(ctypes.byref(token_list))
        
        start = time.perf_counter()
        c_lib.tokenize_fast(data, len(data), ctypes.byref(token_list))
        elapsed = time.perf_counter() - start
        
        c_lib.tokenlist_free(ctypes.byref(token_list))
        times.append(elapsed * 1000)
    
    return times

def bench_rust_regex(text: str, iterations: int) -> list:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        rust_split_text(SPLIT_PATTERN, text)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return times

def stats_summary(times: list) -> dict:
    """Compute statistics from timing list."""
    if not times or len(times) == 0:
        return {}
    
    return {
        'min': min(times),
        'max': max(times),
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
    }

def format_stats(name: str, data_size: int, times: list) -> str:
    """Format timing statistics for output."""
    if not times or len(times) == 0:
        return f"{name:20} {data_size:>10} B  --\n"
    
    stats = stats_summary(times)
    
    return (f"{name:20} {data_size:>10} B  "
            f"min={stats['min']:.3f}ms  max={stats['max']:.3f}ms  "
            f"mean={stats['mean']:.3f}ms  median={stats['median']:.3f}ms  "
            f"stdev={stats['stdev']:.3f}ms\n")

def benchmark_dataset(name: str, data_bytes: bytes, iterations: int) -> None:
    test_text = data_bytes.decode('utf-8', errors='replace')
    
    print(f"\n--- Dataset: {name} ({len(data_bytes)} bytes, {iterations} iterations) ---")
    print()
    
    c_times = bench_c_regex(data_bytes, iterations)
    print(format_stats("C tokenizer", len(data_bytes), c_times), end='')
    
    rust_times = bench_rust_regex(test_text, iterations)
    print(format_stats("Rust split", len(data_bytes), rust_times), end='')
    
    if c_times and rust_times:
        c_mean = statistics.mean(c_times)
        rust_mean = statistics.mean(rust_times)
        ratio = rust_mean / c_mean
        speedup = "C is faster" if ratio > 1 else "Rust is faster"
        print(f"Speedup: {ratio:.2f}x ({speedup})")
    
    print()

    # Verify token splits match between C and Python regex tokenizer
    cmp_text = data_bytes.decode('utf-8', errors='surrogatepass')
    ok, err, out_c, out_py = compare_pair_text(cmp_text)
    if ok:
        print("Compare: OK (C vs Py splits match)")
    else:
        print("Compare: MISMATCH (C vs Py)")
        if err:
            print(err)
        if out_c is not None and out_py is not None:
            c_lines = out_c.splitlines()
            p_lines = out_py.splitlines()
            print(f"C tokens: {len(c_lines)} | Py tokens: {len(p_lines)}")
            print("--- C (head) ---")
            print("\n".join(c_lines[:10]))
            print("--- Py (head) ---")
            print("\n".join(p_lines[:10]))
        # Stop the benchmark if mismatch detected
        raise SystemExit(1)

def main():
    # Check if files were provided as arguments
    file_args = sys.argv[1:] if len(sys.argv) > 1 else []
        
    # If files provided, benchmark them
    if file_args:
        for file_path in file_args:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                continue
            
            try:
                data = path.read_bytes()                
                benchmark_dataset(path.name, data, 10)
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    else:
        # Run random generated data
        configs = [
            ("tiny", 100, 1000),
            ("small", 1024, 500),
            ("medium", 10 * 1024, 100),
            ("large", 100 * 1024, 30),
            ("xlarge", 1024 * 1024, 10),
        ]
        
        for name, size_bytes, iterations in configs:
            # Generate test data
            test_text = gen_valid_unicode_string(
                random.Random(hash(name)), 
                size_bytes
            )
            test_bytes = test_text.encode('utf-8')
            
            benchmark_dataset(name, test_bytes, iterations)
    
    print("=" * 140)


if __name__ == "__main__":
    main()
