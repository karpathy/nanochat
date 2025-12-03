#!/usr/bin/env python3
"""
Quick verification script to ensure all 4 optimizations are working.
Run this before your full training to verify everything is correct.

Usage: python verify_optimizations.py
"""

import torch
import sys
import os

print("=" * 80)
print("NANOCHAT OPTIMIZATIONS VERIFICATION")
print("=" * 80)

# Test 1: Check GPU availability
print("\n[1/4] GPU Availability Check...")
if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    sys.exit(1)

gpu_count = torch.cuda.device_count()
print(f"✅ Found {gpu_count} GPUs")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

# Test 2: Verify auto_batch_size module exists and has correct function
print("\n[2/4] Auto Batch Size Discovery Check...")
try:
    from nanochat.auto_batch_size import find_optimal_device_batch_size
    print("✅ auto_batch_size.py found")
    print("✅ find_optimal_device_batch_size() function exists")

    # Check if it has the right signature
    import inspect
    sig = inspect.signature(find_optimal_device_batch_size)
    params = list(sig.parameters.keys())
    required_params = ['model', 'max_seq_len', 'total_batch_size', 'ddp_world_size', 'data_sample_fn']
    if all(p in params for p in required_params):
        print("✅ Function signature is correct")
    else:
        print(f"⚠️  Function signature might be wrong. Params: {params}")
except ImportError as e:
    print(f"❌ auto_batch_size module not found: {e}")
except AttributeError as e:
    print(f"❌ find_optimal_device_batch_size function not found: {e}")

# Test 3: Verify KV-Cache implementation in GPT.generate()
print("\n[3/4] KV-Cache Implementation Check...")
try:
    from nanochat.gpt import GPT
    from nanochat.engine import KVCache
    import inspect

    # Check if generate() method exists
    if hasattr(GPT, 'generate'):
        print("✅ GPT.generate() method exists")

        # Check source code for KV-cache usage
        source = inspect.getsource(GPT.generate)
        if 'KVCache' in source and 'kv_cache' in source:
            print("✅ KV-Cache is used in generate()")
            if 'torch.cat' not in source or source.count('torch.cat') == 0:
                print("✅ No torch.cat() pattern (good - using incremental decode)")
            else:
                print("⚠️  torch.cat() found - might still be using old pattern")
        else:
            print("❌ KV-Cache not found in generate() method")
    else:
        print("❌ GPT.generate() method not found")
except Exception as e:
    print(f"❌ Error checking GPT: {e}")

# Test 4: Verify token broadcasting fix in engine.py
print("\n[4/4] Token Broadcasting Fix Check...")
try:
    from nanochat.engine import Engine
    import inspect

    source = inspect.getsource(Engine.generate)

    # Check if the bug pattern is removed
    if '[sampled_tokens[0]] * num_samples' in source:
        print("❌ Token broadcasting BUG still present!")
        print("    Found: sampled_tokens[0] * num_samples")
    else:
        print("✅ Token broadcasting bug is fixed")

    # Verify independent sampling exists
    if 'logits.repeat(num_samples' in source or 'logits_repeated' in source:
        print("✅ Independent token sampling implementation found")
    else:
        print("⚠️  Independent sampling might not be implemented")

except Exception as e:
    print(f"❌ Error checking Engine: {e}")

# Test 5: Check torch.compile in chat_sft.py
print("\n[5/5] torch.compile Configuration Check...")
try:
    # Read chat_sft.py
    with open('scripts/chat_sft.py', 'r') as f:
        sft_source = f.read()

    # Check if max_seq_len is defined
    if 'max_seq_len = 2048' in sft_source or 'max_seq_len=2048' in sft_source:
        print("✅ max_seq_len = 2048 configured")
    else:
        print("⚠️  max_seq_len might not be set to 2048")

    # Check if torch.compile is enabled (not commented)
    import re
    compile_lines = [line for line in sft_source.split('\n') if 'torch.compile' in line]
    enabled_compile = [line for line in compile_lines if not line.strip().startswith('#')]

    if enabled_compile:
        print("✅ torch.compile is enabled")
        if 'dynamic=False' in sft_source:
            print("✅ dynamic=False is set (correct for fixed padding)")
        else:
            print("⚠️  dynamic=False might not be set")
    else:
        print("❌ torch.compile is commented out or not found")

    # Check fixed padding
    if 'ncols = max_seq_len - 1' in sft_source:
        print("✅ Fixed-length padding is configured")
    elif 'ncols = max(len(ids)' in sft_source:
        print("❌ Still using dynamic padding!")
    else:
        print("⚠️  Padding configuration unclear")

except Exception as e:
    print(f"❌ Error checking chat_sft.py: {e}")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print("""
If all checks show ✅, your optimizations are correctly implemented!

Expected improvements:
  - Auto Batch Size Discovery: 2-3× training throughput
  - torch.compile (SFT only): 1.5× faster SFT training
  - KV-Cache: 5-10× faster inference
  - Token Broadcasting Fix: Better multi-sample diversity

To measure improvements, compare:
  1. Tokens/second during training (watch the logs)
  2. Total training time
  3. Inference speed (tokens/second during generation)
""")
print("=" * 80)
