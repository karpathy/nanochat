
import os
import sys
import subprocess
import re
import time
import itertools
from typing import Dict, Any, List, Tuple

def run_benchmark(config: Dict[str, Any], env_vars: Dict[str, str], steps: int = 20) -> float:
    """
    Runs a short training session with the given configuration and environment variables.
    Returns the average tokens per second (tok/sec) or -1.0 if failed.
    """

    # Construct command
    cmd = [sys.executable, "-m", "scripts.base_train"]

    # Add config arguments
    # base_train.py uses configurator.py which expects --key=value
    for key, value in config.items():
        cmd.append(f"--{key}={value}")

    # Force low number of iterations for speed
    cmd.append(f"--num_iterations={steps}")
    cmd.append("--run=dummy") # Don't log to wandb

    # Merge environment variables
    current_env = os.environ.copy()
    current_env.update(env_vars)

    print(f"Running benchmark with config: {config} env: {env_vars}")

    try:
        # Capture output to parse tok/sec
        # We need to capture both stdout and stderr as python buffering might mix them
        result = subprocess.run(
            cmd,
            env=current_env,
            capture_output=True,
            text=True,
            timeout=600 # 10 minute timeout per run (increased for compilation)
        )

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}")
            # Check for OOM
            if "OutOfMemoryError" in result.stderr or "OutOfMemoryError" in result.stdout:
                print("Failure reason: OutOfMemoryError")
            else:
                print(f"Stderr tail: {result.stderr[-500:]}")
            return -1.0

        # Parse output for tok/sec
        # Look for lines like: step 00030/00050 ... | tok/sec: 3,200 | ...
        tok_sec_values = []
        # Skip the first few steps as they might be slow (compilation, warmup)
        warmup_steps = 5

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
            print("Could not parse tok/sec from output")
            # Print stdout for debugging
            # print(result.stdout)
            return -1.0

        avg_tok_sec = sum(tok_sec_values) / len(tok_sec_values)
        print(f"Result: {avg_tok_sec:.2f} tok/sec")
        return avg_tok_sec

    except subprocess.TimeoutExpired:
        print("Run timed out")
        return -1.0
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1.0

def main():
    print("Starting System Auto-Tuning...")

    # 1. Hardware Detection (Basic)
    is_rocm = False
    try:
        if os.path.exists("/dev/kfd"):
             is_rocm = True
    except:
        pass

    print(f"Detected Platform: {'ROCm/AMD' if is_rocm else 'CUDA/NVIDIA/CPU'}")

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
                print(f"Batch size {bs} failed, stopping search for this env config.")
                break

    print("\n" + "="*40)
    print("Tuning Results:")
    print("="*40)

    if not results:
        print("No successful runs found.")
        sys.exit(1)

    # Sort by throughput
    results.sort(key=lambda x: x[2], reverse=True)

    for conf, env, tp in results:
        env_str = " ".join([f"{k}={v}" for k,v in env.items()]) if env else "Default Env"
        print(f"Throughput: {tp:,.2f} tok/sec | BS: {conf['device_batch_size']} | Env: {env_str}")

    print("\n" + "="*40)
    print("Best Configuration:")
    print("="*40)
    print(f"Throughput: {best_throughput:,.2f} tok/sec")
    print("Config Parameters:")
    for k, v in best_config.items():
        print(f"  --{k}={v}")

    if best_env:
        print("Environment Variables:")
        for k, v in best_env.items():
            print(f"  export {k}={v}")

    # Generate a suggestion string for speedrun.sh
    print("\nSuggestion for speedrun.sh modifications:")
    print(f"export HSA_OVERRIDE_GFX_VERSION=11.5.1 # (Ensure this matches your hardware)")
    if best_env:
        for k, v in best_env.items():
            print(f"export {k}={v}")

    cmd_args = " ".join([f"--{k}={v}" for k,v in best_config.items()])
    print(f"python -m scripts.base_train {cmd_args} --run=$WANDB_RUN")

if __name__ == "__main__":
    main()
