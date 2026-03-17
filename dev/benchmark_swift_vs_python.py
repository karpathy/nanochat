"""
Benchmark: Swift MLX stub (KV-cache, GPU) vs Python MLX forward (no KV-cache, GPU).

Measures per-token decode latency on the same d4 checkpoint to establish the
latency baseline comparison for Story 4a.
"""

import json
import time
import subprocess
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
if not (REPO / "nanochat").exists():
    REPO = Path(os.environ.get("REPO", "/Users/peternicholls/Dev/nanochatter"))

MANIFEST_PATH = REPO / "runs" / "mlx_exports" / "phase2_d4_l_mps_step20.json"
PROMPT_TOKENS = [32759, 483, 2027, 5636, 286, 668, 306]  # "The chemical formula of water is"
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 2
TIMED_RUNS = 5

# ---------------------------------------------------------------------------
# Python MLX model (minimal, no KV-cache)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO))
from dev.mlx_gpt_prototype import MLXGPTPrototype, MLXGPTConfig


def load_python_model(manifest_path: Path):
    """Load the MLX model from the exported safetensors checkpoint."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    config = manifest["config"]
    safetensors_rel = manifest["export"]["safetensors_path"]
    safetensors_path = manifest_path.parent / Path(safetensors_rel).name

    mlx_config = MLXGPTConfig(
        sequence_len=config["sequence_len"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_head=config["n_kv_head"],
        n_embd=config["n_embd"],
        window_pattern=config["window_pattern"],
    )
    model = MLXGPTPrototype(mlx_config)

    # Load safetensors weights using mx.load
    weights = mx.load(str(safetensors_path))
    # Map flat tensor names to the nn.Module tree expected by load_weights
    weight_list = list(weights.items())
    model.load_weights(weight_list)
    mx.eval(model.parameters())
    return model


def python_mlx_greedy_generate(model, prompt_tokens, max_new_tokens):
    """Full-prefix recompute greedy generation (no KV-cache)."""
    all_tokens = list(prompt_tokens)
    decode_times = []

    # First token (prefill)
    idx = mx.array([all_tokens], dtype=mx.int32)
    logits = model(idx)
    mx.eval(logits)
    next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    all_tokens.append(next_id)

    for _ in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        idx = mx.array([all_tokens], dtype=mx.int32)
        logits = model(idx)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000.0)
        all_tokens.append(next_id)

    return all_tokens[len(prompt_tokens):], decode_times


# ---------------------------------------------------------------------------
# Swift MLX stub benchmark
# ---------------------------------------------------------------------------
def swift_mlx_generate(manifest_path, prompt_tokens, max_new_tokens, device="gpu"):
    """Run the Swift stub and parse timing from its output."""
    binary = REPO / "swift" / "Build" / "Products" / "Debug" / "nanochat-mlx-stub"
    env = os.environ.copy()
    env["DYLD_FRAMEWORK_PATH"] = str(REPO / "swift" / "Build" / "Products" / "Debug")
    token_arg = ",".join(str(t) for t in prompt_tokens)
    cmd = [
        str(binary), "--manifest", str(manifest_path),
        "--prompt-tokens", token_arg,
        "--max-new-tokens", str(max_new_tokens),
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
    if result.returncode != 0:
        print("Swift stub error:", result.stderr, file=sys.stderr)
        return None
    timing = {}
    for line in result.stdout.splitlines():
        if line.startswith("Timing: "):
            for pair in line[len("Timing: "):].split():
                k, _, v = pair.partition("=")
                timing[k] = v
        if line.startswith("Generated token ids: "):
            payload = line[len("Generated token ids: "):].strip()
            timing["tokens"] = [int(t) for t in payload.split(",") if t]
    return timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Benchmark: Swift MLX (KV-cache, GPU) vs Python MLX (no KV-cache, GPU)")
    print(f"Checkpoint: {MANIFEST_PATH.name}")
    print(f"Prompt tokens: {len(PROMPT_TOKENS)}, max new tokens: {MAX_NEW_TOKENS}")
    print()

    # --- Python MLX ---
    print("Loading Python MLX model...")
    model = load_python_model(MANIFEST_PATH)
    print(f"Model params: {model.num_params():,}")

    # Warmup
    for i in range(WARMUP_RUNS):
        tokens, _ = python_mlx_greedy_generate(model, PROMPT_TOKENS, MAX_NEW_TOKENS)
        print(f"  warmup {i+1}: first token = {tokens[0]}")

    # Timed runs
    python_decode_times_all = []
    for i in range(TIMED_RUNS):
        tokens, decode_times = python_mlx_greedy_generate(model, PROMPT_TOKENS, MAX_NEW_TOKENS)
        avg = sum(decode_times) / len(decode_times) if decode_times else 0
        python_decode_times_all.append(avg)
        print(f"  run {i+1}: avg_decode={avg:.2f}ms ({len(decode_times)} steps)")

    python_avg = sum(python_decode_times_all) / len(python_decode_times_all)
    print(f"\nPython MLX avg decode: {python_avg:.2f}ms/token (no KV-cache, full recompute)")
    print(f"  First generated token: {tokens[0]}")
    print()

    # --- Swift MLX GPU ---
    print("Running Swift MLX stub (GPU, KV-cache)...")

    # Warmup
    for i in range(WARMUP_RUNS):
        t = swift_mlx_generate(MANIFEST_PATH, PROMPT_TOKENS, MAX_NEW_TOKENS, "gpu")
        if t:
            print(f"  warmup {i+1}: avg_decode={t.get('avg_decode', '?')}")

    # Timed runs
    swift_decode_times = []
    swift_prefill_times = []
    for i in range(TIMED_RUNS):
        t = swift_mlx_generate(MANIFEST_PATH, PROMPT_TOKENS, MAX_NEW_TOKENS, "gpu")
        if t:
            avg_str = t.get("avg_decode", "0")
            avg_val = float(avg_str.replace("ms", ""))
            swift_decode_times.append(avg_val)
            prefill_str = t.get("prefill", "0")
            prefill_val = float(prefill_str.replace("ms", ""))
            swift_prefill_times.append(prefill_val)
            print(f"  run {i+1}: prefill={prefill_str} avg_decode={avg_str}")

    swift_avg = sum(swift_decode_times) / len(swift_decode_times) if swift_decode_times else 0
    swift_prefill_avg = sum(swift_prefill_times) / len(swift_prefill_times) if swift_prefill_times else 0
    print(f"\nSwift MLX avg decode: {swift_avg:.2f}ms/token (KV-cache, GPU)")
    print(f"Swift MLX avg prefill: {swift_prefill_avg:.1f}ms")
    if t:
        print(f"  First generated token: {t.get('tokens', [None])[0] if t.get('tokens') else '?'}")
    print()

    # --- Swift MLX CPU ---
    print("Running Swift MLX stub (CPU, KV-cache)...")
    swift_cpu_times = []
    for i in range(TIMED_RUNS):
        t = swift_mlx_generate(MANIFEST_PATH, PROMPT_TOKENS, MAX_NEW_TOKENS, "cpu")
        if t:
            avg_str = t.get("avg_decode", "0")
            avg_val = float(avg_str.replace("ms", ""))
            swift_cpu_times.append(avg_val)
            print(f"  run {i+1}: avg_decode={avg_str}")

    swift_cpu_avg = sum(swift_cpu_times) / len(swift_cpu_times) if swift_cpu_times else 0
    print(f"\nSwift MLX CPU avg decode: {swift_cpu_avg:.2f}ms/token (KV-cache, CPU)")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY (d4 model, 4 layers, ~2M params)")
    print("=" * 60)
    print(f"Python MLX (no KV-cache, GPU default): {python_avg:.2f}ms/token")
    print(f"Swift  MLX (KV-cache, GPU):             {swift_avg:.2f}ms/token")
    print(f"Swift  MLX (KV-cache, CPU):             {swift_cpu_avg:.2f}ms/token")
    if swift_avg > 0:
        print(f"Speedup (Swift GPU vs Python):          {python_avg / swift_avg:.1f}x")
    print()
    print("NOTE: This is a tiny d4 model. At 2.8B params, the GPU work")
    print("per token is O(10-30ms) and the Python overhead is O(1-5ms),")
    print("so the KV-cache + Swift path should show clearer wins.")


if __name__ == "__main__":
    main()
