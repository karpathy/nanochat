"""Benchmark the persistent Swift MLX worker (serve-stdin mode) for multi-request latency.

This script measures steady-state per-token decode latency when reusing a single
Swift worker process across multiple requests, so the model stays loaded between calls.

Baseline (d32, M2 Ultra, single-shot path):
  Python MLX (no KV-cache)  : 27.79 ms/token
  Swift one-shot (KV-cache) : 30.29 ms/token

Usage:
  python dev/benchmark_persistent_worker.py --manifest runs/mlx_exports/mlx_reference_d32.json \\
    --prompt "The chemical formula of water is" --max-new-tokens 32 --timed-runs 8
"""

import argparse
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nanochat.tokenizer import get_tokenizer
from nanochat.swift_stub_engine import SwiftStubEngine

# ---------------------------------------------------------------------------
# Known baselines (d32, M2 Ultra)
# ---------------------------------------------------------------------------
BASELINES = {
    "mlx_reference_d32.json": {
        "python_mlx_no_kvcache_ms": 27.79,
        "swift_oneshot_gpu_ms": 30.29,
    }
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent Swift worker multi-request benchmark")
    p.add_argument("--manifest", default="runs/mlx_exports/mlx_reference_d32.json")
    p.add_argument("--prompt", default="The chemical formula of water is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup-runs", type=int, default=2,
                   help="Requests to discard (model warm-up)")
    p.add_argument("--timed-runs", type=int, default=8,
                   help="Requests to include in timing statistics")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    return p.parse_args()


def ms(s: str) -> float:
    """Parse a timing string like '27.8ms' or '27.8' to float ms."""
    return float(s.replace("ms", "").strip())


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    prompt_tokens = tokenizer.encode(args.prompt, prepend=bos)

    print("=" * 60)
    print("Persistent Swift Worker Benchmark")
    print("=" * 60)
    print(f"Manifest   : {manifest_path.name}")
    print(f"Prompt     : '{args.prompt}'")
    print(f"Tokens in  : {len(prompt_tokens)}")
    print(f"Max new    : {args.max_new_tokens}")
    print(f"Device     : {args.device}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Timed runs : {args.timed_runs}")
    print()

    engine = SwiftStubEngine(
        tokenizer=tokenizer,
        manifest_path=str(manifest_path),
        device=args.device,
    )

    total_runs = args.warmup_runs + args.timed_runs
    results: list[dict] = []

    print(f"Running {total_runs} requests ({args.warmup_runs} warmup + {args.timed_runs} timed)...")
    print()

    for i in range(total_runs):
        generated = list(engine.generate(prompt_tokens, max_tokens=args.max_new_tokens))
        timing = engine.last_timing or {}
        is_warmup = i < args.warmup_runs
        label = f"warmup {i + 1}" if is_warmup else f"run    {i + 1 - args.warmup_runs}"

        prefill_ms = ms(timing.get("prefill", "0")) if timing else 0.0
        avg_decode_ms = ms(timing.get("avg_decode", "0")) if timing else 0.0
        tokens_decoded = timing.get("tokens_decoded", "?")

        print(
            f"  {label}: prefill={prefill_ms:.1f}ms  "
            f"avg_decode={avg_decode_ms:.2f}ms  "
            f"tokens={tokens_decoded}  "
            f"device={timing.get('device', '?')}"
        )

        if not is_warmup:
            results.append({
                "prefill_ms": prefill_ms,
                "avg_decode_ms": avg_decode_ms,
            })

    engine.close()

    if not results:
        print("No timed results collected.")
        return

    prefill_list = [r["prefill_ms"] for r in results]
    decode_list = [r["avg_decode_ms"] for r in results]

    avg_prefill = sum(prefill_list) / len(prefill_list)
    avg_decode = sum(decode_list) / len(decode_list)
    min_decode = min(decode_list)
    max_decode = max(decode_list)

    manifest_key = manifest_path.name
    baseline = BASELINES.get(manifest_key, {})

    print()
    print("=" * 60)
    print("RESULTS — Persistent worker (model stays loaded)")
    print("=" * 60)
    print(f"  avg prefill    : {avg_prefill:.1f} ms")
    print(f"  avg decode     : {avg_decode:.2f} ms/token")
    print(f"  min decode     : {min_decode:.2f} ms/token")
    print(f"  max decode     : {max_decode:.2f} ms/token")
    print()

    if baseline:
        py_base = baseline.get("python_mlx_no_kvcache_ms")
        sw_base = baseline.get("swift_oneshot_gpu_ms")
        print("COMPARISON vs baselines (d32, M2 Ultra)")
        print(f"  Python MLX  (no KV-cache, one-shot) : {py_base:.2f} ms/token")
        print(f"  Swift GPU   (KV-cache,    one-shot) : {sw_base:.2f} ms/token")
        print(f"  Swift GPU   (KV-cache,  persistent) : {avg_decode:.2f} ms/token  ← this run")
        print()
        if py_base:
            speedup_vs_python = py_base / avg_decode if avg_decode > 0 else 0.0
            delta_vs_python = avg_decode - py_base
            symbol = "faster" if speedup_vs_python > 1.0 else "slower"
            print(f"  Persistent worker vs Python MLX : {speedup_vs_python:.2f}x ({abs(delta_vs_python):.2f}ms {symbol})")
        if sw_base:
            speedup_vs_oneshot = sw_base / avg_decode if avg_decode > 0 else 0.0
            delta_vs_oneshot = avg_decode - sw_base
            symbol = "faster" if speedup_vs_oneshot > 1.0 else "slower"
            print(f"  Persistent worker vs one-shot   : {speedup_vs_oneshot:.2f}x ({abs(delta_vs_oneshot):.2f}ms {symbol})")

    print("=" * 60)


if __name__ == "__main__":
    main()
