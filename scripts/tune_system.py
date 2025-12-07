
import os
import sys
import subprocess
import re
import time
import shutil
import json
import argparse
from typing import Dict, Any, List, Tuple

def run_benchmark(config_overrides: Dict[str, Any], env_vars: Dict[str, str], base_config_path: str = None, steps: int = 5, minimal_validation: bool = True) -> float:
    """
    Runs a short training session with the given configuration and environment variables.
    Returns the average tokens per second (tok/sec) or -1.0 if failed.
    """

    # Construct command
    # Use -u for unbuffered output to ensure we capture stdout even if it hangs/crashes
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"]

    # Pass the base config file if provided
    # This allows base_train to safely load all keys (including unknown ones) from the JSON
    if base_config_path:
        cmd.append(base_config_path)

    # Add config overrides as flags
    # These will override values from the base config file
    for key, value in config_overrides.items():
        cmd.append(f"--{key}={value}")

    # Force low number of iterations for speed
    cmd.append(f"--num_iterations={steps}")
    cmd.append("--run=dummy") # Don't log to wandb
    cmd.append("--core_metric_every=-1") # Disable heavy evaluation
    cmd.append("--save_every=-1") # Disable intermediate checkpointing

    # Optimize validation overhead:
    # If minimal_validation is True, reduce eval_tokens to a small multiple of the batch size
    # to ensure validation is near-instant, preventing timeouts on small batch sizes.
    if minimal_validation:
        bs = int(config_overrides.get("device_batch_size", 16))

        # We need max_seq_len to calculate eval_tokens.
        # If it's not in overrides, we need to read it from base_config_path or assume default.
        seq_len = 2048 # default
        if "max_seq_len" in config_overrides:
             seq_len = int(config_overrides["max_seq_len"])
        elif base_config_path:
            try:
                with open(base_config_path) as f:
                    base_conf = json.load(f)
                    seq_len = int(base_conf.get("max_seq_len", 2048))
            except:
                pass

        eval_tokens = bs * seq_len * 2
        cmd.append(f"--eval_tokens={eval_tokens}")

    # Merge environment variables
    current_env = os.environ.copy()
    current_env.update(env_vars)

    print(f"Running benchmark with overrides: {config_overrides} env: {env_vars}", flush=True)

    try:
        # Capture output to parse tok/sec
        result = subprocess.run(
            cmd,
            env=current_env,
            capture_output=True,
            text=True,
            timeout=1200 # 20 minute timeout per run
        )

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}", flush=True)
            # Check for OOM
            if "OutOfMemoryError" in result.stderr or "OutOfMemoryError" in result.stdout:
                print("Failure reason: OutOfMemoryError", flush=True)
            else:
                print(f"Stderr tail: {result.stderr[-5000:]}", flush=True)
            return -1.0

        # Parse output for tok/sec
        tok_sec_values = []
        # Skip the first few steps as they might be slow (compilation, warmup)
        warmup_steps = 2

        for line in result.stdout.splitlines():
            match = re.search(r"tok/sec:\s*([\d,]+)", line)
            if match:
                step_match = re.search(r"step\s+(\d+)", line)
                if step_match:
                    step = int(step_match.group(1))
                    if step > warmup_steps:
                        val = float(match.group(1).replace(',', ''))
                        tok_sec_values.append(val)

        if not tok_sec_values:
            print("Could not parse tok/sec from output", flush=True)
            return -1.0

        avg_tok_sec = sum(tok_sec_values) / len(tok_sec_values)
        print(f"Result: {avg_tok_sec:.2f} tok/sec", flush=True)
        return avg_tok_sec

    except subprocess.TimeoutExpired as e:
        print(f"Run timed out after {e.timeout} seconds", flush=True)
        print("--- Stdout during timeout ---", flush=True)
        if e.stdout:
            print(e.stdout, flush=True)
        else:
            print("(No stdout captured)", flush=True)

        print("--- Stderr during timeout ---", flush=True)
        if e.stderr:
            print(e.stderr, flush=True)
        else:
            print("(No stderr captured)", flush=True)

        return -1.0
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return -1.0

def main():
    print("Starting System Auto-Tuning...", flush=True)

    parser = argparse.ArgumentParser(description="Auto-tune system performance")
    parser.add_argument("--config", type=str, default=None, help="Path to base JSON configuration file")
    args = parser.parse_args()

    # Load Base Configuration for reference (but don't rely on it for cmd construction unless needed)
    base_config = {}
    if args.config:
        print(f"Loading base configuration from {args.config}", flush=True)
        try:
            with open(args.config) as f:
                base_config = json.load(f)
        except Exception as e:
             print(f"Error loading config file: {e}", flush=True)
             sys.exit(1)
    else:
        print("Using default configuration (Depth 10)", flush=True)
        base_config = {"depth": 10, "max_seq_len": 2048}

    # 1. Hardware Detection (Basic)
    is_rocm = False
    try:
        if os.path.exists("/dev/kfd"):
             is_rocm = True
    except:
        pass

    # Check for Strix Halo (gfx1151)
    is_strix_halo = False
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.5.1":
        is_strix_halo = True
    elif shutil.which('rocminfo'):
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            if 'gfx1151' in result.stdout:
                is_strix_halo = True
        except:
            pass

    print(f"Detected Platform: {'ROCm/AMD' if is_rocm else 'CUDA/NVIDIA/CPU'}", flush=True)
    if is_strix_halo:
        print("Detected Variant: Strix Halo (gfx1151)", flush=True)

    # 2. Define Search Space
    # Batch sizes to try. We start small and go up.
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Compilation flags
    compile_options = [True, False] if is_rocm else [True]

    if is_strix_halo:
        print("NOTE: Disabling 'compile=True' for tuning on Strix Halo due to known stability issues.", flush=True)
        compile_options = [False]

    # Environment variable combinations
    env_configs = [{}]
    if is_rocm:
        # Tuning ROCm specific flags
        env_configs = [
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"},
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
        ]

    MINIMAL_VALIDATION = True
    if MINIMAL_VALIDATION:
        print("NOTE: Minimal validation enabled (eval_tokens reduced) to prevent timeouts.", flush=True)

    best_throughput = 0.0
    best_overrides = None
    best_env = None

    results = []

    # Grid search for Throughput
    print("\nPhase 1: Throughput Tuning (Batch Size & Compilation)", flush=True)

    for env_vars in env_configs:
        for compile_opt in compile_options:
            for bs in batch_sizes:
                # Construct overrides
                overrides = {
                    "device_batch_size": bs,
                    "depth": depth,
                    "compile": str(compile_opt),
                    "eval_tokens": bs * 2048, # Scale validation to avoid timeout (1 step)
                }

                throughput = run_benchmark(overrides, env_vars, base_config_path=args.config, minimal_validation=MINIMAL_VALIDATION)

                if throughput > 0:
                    results.append((overrides, env_vars, throughput))
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_overrides = overrides
                        best_env = env_vars
                else:
                    # If we failed (likely OOM), larger batch sizes will likely also fail
                    # So break the inner loop
                    print(f"Batch size {bs} failed, stopping search for this env config.", flush=True)
                    break

    print("\n" + "="*40, flush=True)
    print("Tuning Results:", flush=True)
    print("="*40, flush=True)

    if not results:
        print("No successful runs found.", flush=True)
        sys.exit(1)

    # Sort by throughput
    results.sort(key=lambda x: x[2], reverse=True)

    for ovr, env, tp in results:
        env_str = " ".join([f"{k}={v}" for k,v in env.items()]) if env else "Default Env"
        print(f"Throughput: {tp:,.2f} tok/sec | BS: {ovr['device_batch_size']} | Compile: {ovr['compile']} | Env: {env_str}", flush=True)

    print("\n" + "="*40, flush=True)
    print("Best Throughput Configuration:", flush=True)
    print("="*40, flush=True)
    print(f"Throughput: {best_throughput:,.2f} tok/sec", flush=True)

    print("\nRecommended Updated Configuration:", flush=True)
    print("You can update your config file with these values:")
    print("-" * 20)

    # Create a merged config for display
    final_config = base_config.copy()
    if best_overrides:
        final_config.update(best_overrides)
        # Type conversion for JSON
        if "compile" in final_config and final_config["compile"] == "True": final_config["compile"] = True
        if "compile" in final_config and final_config["compile"] == "False": final_config["compile"] = False

    print(json.dumps(final_config, indent=4), flush=True)
    print("-" * 20)

    if best_env:
        print("Recommended Environment Variables:", flush=True)
        for k, v in best_env.items():
            print(f"  export {k}={v}", flush=True)

    # Command line suggestion
    cmd_args = " ".join([f"--{k}={v}" for k,v in final_config.items()])
    # Note: final_config might contain keys that are not flags but are from the json (like max_chars)
    # But for a direct python -m run, we'd probably want to use the config file + overrides.

    print(f"\nRun command with updated profile:\npython -m scripts.base_train {args.config if args.config else ''} --device_batch_size={best_overrides['device_batch_size']} --compile={best_overrides['compile']} --run=$WANDB_RUN", flush=True)

if __name__ == "__main__":
    main()
