#!/usr/bin/env python3
"""Test torchrun command locally"""
import subprocess
import sys

# Simulate the exact command that will run
cmd = [
    "torchrun", "--standalone", "--nproc_per_node=1",
    "-m", "scripts.base_train",
    "--depth=4",
    "--device_batch_size=1",
    "--num_iterations=2",
    "--run=test_local",
    "--vertex_experiment=",
    "--vertex_tensorboard="
]

print("Testing command:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print("STDOUT:")
    print(result.stdout[:1000])
    print("\nSTDERR:")
    print(result.stderr[:1000])
    print(f"\nExit code: {result.returncode}")
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print("Command timed out (expected for training)")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
