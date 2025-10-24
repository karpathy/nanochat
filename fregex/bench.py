import sys
import ctypes
import random
import time
import statistics
import os
import gc
from pathlib import Path

from nanochat.tokenizer import SPLIT_PATTERN

os.environ.update({
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'RAYON_NUM_THREADS': '1',
})

os.setpriority(os.PRIO_PROCESS, 0, -10)

from rustbpe import split_text as rust_split_text
from fregex.fuzz import gen_valid_unicode_string, compare_pair_text
from fregex.cload import *

PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
PyBytes_AsString.restype = ctypes.c_void_p
PyBytes_AsString.argtypes = [ctypes.py_object]

def _run_once_c(data: bytes) -> float:
    token_list = TokenList()
    c_lib.tokenlist_init(ctypes.byref(token_list))
    base_ptr = PyBytes_AsString(data)
    t0 = time.perf_counter_ns()
    c_lib.tokenize_fast(base_ptr, len(data), ctypes.byref(token_list))
    dt_ms = (time.perf_counter_ns() - t0) / 1e6
    c_lib.tokenlist_free(ctypes.byref(token_list))
    return dt_ms

def _run_once_rust(text: str) -> float:
    t0 = time.perf_counter_ns()
    rust_split_text(SPLIT_PATTERN, text)
    return (time.perf_counter_ns() - t0) / 1e6

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

    # Pre-touch data to avoid first-touch/page-fault skew
    if data_bytes:
        _ = data_bytes[0]
        for i in range(0, len(data_bytes), 4096):
            _ = data_bytes[i]

    # Warm-up
    for _ in range(20):
        _run_once_c(data_bytes)
        _run_once_rust(test_text)

    # Disable GC during timed section
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    c_times = []
    rust_times = []
    for _ in range(iterations):
        c_times.append(_run_once_c(data_bytes))
        rust_times.append(_run_once_rust(test_text))

    if gc_was_enabled:
        gc.enable()

    print(format_stats("C tokenizer", len(data_bytes), c_times), end='')
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
                benchmark_dataset(path.name, data, 1_000)
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    else:
        # Run random generated data
        configs = [
            ("tiny", 100, 1000),
            ("small", 1024, 500),
            ("medium", 10 * 1024, 100),
            ("large", 100 * 1024, 100),
            ("xlarge", 1024 * 1024, 100),
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
