"""Benchmark Swift MLX stub against the Python MLX prototype on exported checkpoints."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parent
if not (REPO / "nanochat").exists():
    REPO = Path(os.environ.get("REPO", "/Users/peternicholls/Dev/nanochatter"))

# ---------------------------------------------------------------------------
# Python MLX model (minimal, no KV-cache)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO))
from dev.mlx_gpt_prototype import MLXGPTPrototype, MLXGPTConfig
from nanochat.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Swift MLX stub against Python MLX on an exported checkpoint")
    parser.add_argument("--manifest", type=str, default="runs/mlx_exports/phase2_d4_l_mps_step20.json")
    parser.add_argument("--prompt", type=str, default="The chemical formula of water is")
    parser.add_argument("--prompt-tokens", type=str, default=None, help="Optional comma-separated token ids to bypass Python tokenization")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--swift-device", type=str, choices=["gpu", "cpu", "both"], default="both")
    parser.add_argument("--skip-python", action="store_true")
    return parser.parse_args()


def resolve_repo_path(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return REPO / path


def build_prompt_tokens(args: argparse.Namespace) -> list[int]:
    if args.prompt_tokens is not None:
        return [int(token) for token in args.prompt_tokens.split(",") if token.strip()]

    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    return tokenizer.encode(args.prompt, prepend=bos_token_id)


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
    args = parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    prompt_tokens = build_prompt_tokens(args)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Benchmark: Swift MLX stub vs Python MLX prototype")
    print(f"Checkpoint: {manifest_path.name}")
    print(
        "Model config: "
        f"d{manifest['config']['n_layer']} "
        f"emb={manifest['config']['n_embd']} "
        f"heads={manifest['config']['n_head']} "
        f"vocab={manifest['config']['vocab_size']}"
    )
    print(f"Prompt tokens: {len(prompt_tokens)}, max new tokens: {args.max_new_tokens}")
    print()

    python_avg = None
    if not args.skip_python:
        print("Loading Python MLX model...")
        model = load_python_model(manifest_path)
        print(f"Model params: {model.num_params():,}")

        for i in range(args.warmup_runs):
            tokens, _ = python_mlx_greedy_generate(model, prompt_tokens, args.max_new_tokens)
            print(f"  warmup {i + 1}: first token = {tokens[0]}")

        python_decode_times_all = []
        for i in range(args.timed_runs):
            tokens, decode_times = python_mlx_greedy_generate(model, prompt_tokens, args.max_new_tokens)
            avg = sum(decode_times) / len(decode_times) if decode_times else 0
            python_decode_times_all.append(avg)
            print(f"  run {i + 1}: avg_decode={avg:.2f}ms ({len(decode_times)} steps)")

        python_avg = sum(python_decode_times_all) / len(python_decode_times_all)
        print(f"\nPython MLX avg decode: {python_avg:.2f}ms/token (no KV-cache, full recompute)")
        print(f"  First generated token: {tokens[0]}")
        print()

    def run_swift_device(device: str) -> tuple[float, float, dict[str, object] | None]:
        print(f"Running Swift MLX stub ({device.upper()}, KV-cache)...")
        last_timing = None
        for i in range(args.warmup_runs):
            timing = swift_mlx_generate(manifest_path, prompt_tokens, args.max_new_tokens, device)
            if timing:
                print(f"  warmup {i + 1}: avg_decode={timing.get('avg_decode', '?')}")

        swift_decode_times = []
        swift_prefill_times = []
        for i in range(args.timed_runs):
            timing = swift_mlx_generate(manifest_path, prompt_tokens, args.max_new_tokens, device)
            if timing:
                last_timing = timing
                avg_val = float(timing.get("avg_decode", "0").replace("ms", ""))
                prefill_val = float(timing.get("prefill", "0").replace("ms", ""))
                swift_decode_times.append(avg_val)
                swift_prefill_times.append(prefill_val)
                print(f"  run {i + 1}: prefill={timing.get('prefill', '?')} avg_decode={timing.get('avg_decode', '?')}")

        decode_avg = sum(swift_decode_times) / len(swift_decode_times) if swift_decode_times else 0.0
        prefill_avg = sum(swift_prefill_times) / len(swift_prefill_times) if swift_prefill_times else 0.0
        print(f"\nSwift MLX avg decode: {decode_avg:.2f}ms/token ({device.upper()}, KV-cache)")
        print(f"Swift MLX avg prefill: {prefill_avg:.1f}ms")
        if last_timing:
            print(f"  First generated token: {last_timing.get('tokens', [None])[0] if last_timing.get('tokens') else '?'}")
        print()
        return decode_avg, prefill_avg, last_timing

    swift_gpu_avg = None
    swift_cpu_avg = None
    if args.swift_device in {"gpu", "both"}:
        swift_gpu_avg, _, _ = run_swift_device("gpu")
    if args.swift_device in {"cpu", "both"}:
        swift_cpu_avg, _, _ = run_swift_device("cpu")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if python_avg is not None:
        print(f"Python MLX (no KV-cache, GPU default): {python_avg:.2f}ms/token")
    if swift_gpu_avg is not None:
        print(f"Swift  MLX (KV-cache, GPU):             {swift_gpu_avg:.2f}ms/token")
    if swift_cpu_avg is not None:
        print(f"Swift  MLX (KV-cache, CPU):             {swift_cpu_avg:.2f}ms/token")
    if python_avg is not None and swift_gpu_avg not in {None, 0.0}:
        print(f"Speedup (Swift GPU vs Python):          {python_avg / swift_gpu_avg:.1f}x")


if __name__ == "__main__":
    main()
