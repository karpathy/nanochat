
import os
import sys
import subprocess
import re
import time
import itertools
from typing import Dict, Any, List, Tuple

def run_benchmark(config: Dict[str, Any], env_vars: Dict[str, str], steps: int = 5) -> float:
    """
    Runs a short training session with the given configuration and environment variables.
    Returns the average tokens per second (tok/sec) or -1.0 if failed.
    """

    # Construct command
    # Use -u for unbuffered output to ensure we capture stdout even if it hangs/crashes
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"]

    # Add config arguments
    # base_train.py uses configurator.py which expects --key=value
    for key, value in config.items():
        cmd.append(f"--{key}={value}")

    # Force low number of iterations for speed
    cmd.append(f"--num_iterations={steps}")
    cmd.append("--run=dummy") # Don't log to wandb

    # Merge environment variables
    current_env = os.environ.copy()
    # Enable torch.compile debugging to stderr
    current_env["TORCH_LOGS"] = "+dynamo"
    current_env.update(env_vars)

    print(f"Running benchmark with config: {config} env: {env_vars}", flush=True)

    try:
        # Capture output to parse tok/sec
        # We need to capture both stdout and stderr as python buffering might mix them
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
                print(f"Stderr tail: {result.stderr[-2000:]}", flush=True)
                # print(f"Stdout tail: {result.stdout[-2000:]}", flush=True)
            return -1.0

        # Parse output for tok/sec
        # Look for lines like: step 00030/00050 ... | tok/sec: 3,200 | ...
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
            print(f"Stdout dump:\n{result.stdout}", flush=True)
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

    # 1. Hardware Detection (Basic)
    is_rocm = False
    try:
        if os.path.exists("/dev/kfd"):
             is_rocm = True
    except:
        pass

    print(f"Detected Platform: {'ROCm/AMD' if is_rocm else 'CUDA/NVIDIA/CPU'}", flush=True)

    # 2. Define Search Space

    # Batch sizes to try. We start small and go up.
    batch_sizes = [16, 32, 64, 128, 256]

    # Model depths to consider (affects throughput heavily)
    # If the user wants to "speedrun", smaller is better, but let's see what fits.
    # We will fix depth to what is likely desired or just benchmark batch size for a fixed depth.
    # Let's try to optimize batch size for the default depth first, or allow user to specify?
    # For now, let's stick to the Speedrun default of depth=10 for the tuning,
    # or maybe we want to see if we can handle depth=20 with smaller batch sizes.
    # Let's just tune batch size and compilation flags for a fixed depth (e.g. 10).
    depth = 10

    # Environment variable combinations
    env_configs = [{}]
    if is_rocm:
        env_configs = [
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"},
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
        ]

    best_throughput = 0.0
    best_config = None
    best_env = None

    results = []

    for env_vars in env_configs:
        for bs in batch_sizes:
            config = {
                "device_batch_size": bs,
                "depth": depth
            }

            throughput = run_benchmark(config, env_vars)

            if throughput > 0:
                results.append((config, env_vars, throughput))
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
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

    for conf, env, tp in results:
        env_str = " ".join([f"{k}={v}" for k,v in env.items()]) if env else "Default Env"
        print(f"Throughput: {tp:,.2f} tok/sec | BS: {conf['device_batch_size']} | Env: {env_str}", flush=True)

    print("\n" + "="*40, flush=True)
    print("Best Configuration:", flush=True)
    print("="*40, flush=True)
    print(f"Throughput: {best_throughput:,.2f} tok/sec", flush=True)
    print("Config Parameters:", flush=True)
    for k, v in best_config.items():
        print(f"  --{k}={v}", flush=True)

    if best_env:
        print("Environment Variables:", flush=True)
        for k, v in best_env.items():
            print(f"  export {k}={v}", flush=True)

    # Generate a suggestion string for speedrun.sh
    print("\nSuggestion for speedrun.sh modifications:", flush=True)
    print(f"export HSA_OVERRIDE_GFX_VERSION=11.5.1 # (Ensure this matches your hardware)", flush=True)
    if best_env:
        for k, v in best_env.items():
            print(f"export {k}={v}", flush=True)

    cmd_args = " ".join([f"--{k}={v}" for k,v in best_config.items()])
    print(f"python -m scripts.base_train {cmd_args} --run=$WANDB_RUN", flush=True)

if __name__ == "__main__":
    main()
