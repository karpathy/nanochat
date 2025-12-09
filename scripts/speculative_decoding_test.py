#!/usr/bin/env python
"""
Test speculative decoding with d4 as draft and attn_mha as target
"""
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
import time

# Initialize
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
device_type = autodetect_device_type()

# Load models
print("Loading draft model (d4)...")
draft_model, tokenizer, draft_meta = load_model(
    "base", device, phase="eval", model_tag="d4"
)
print(f"  - {draft_meta['model_config']['n_layer']} layers")
print(f"  - {sum(p.numel() for p in draft_model.parameters()):,} parameters")

print("\nLoading target model (attn_mha)...")
target_model, _, target_meta = load_model(
    "base", device, phase="eval", model_tag="attn_mha"
)
print(f"  - {target_meta['model_config']['n_layer']} layers")
print(f"  - {sum(p.numel() for p in target_model.parameters()):,} parameters")

# Verify compatibility
assert draft_model.config.vocab_size == target_model.config.vocab_size, \
    "vocab size must be the same!"
print(f"\n✅ Models compatible (vocab_size={draft_model.config.vocab_size})")

# Test generation
engine = Engine(target_model, tokenizer)
bos = tokenizer.get_bos_token_id()
prompt = "Once upon a time in a land far away"
prompt_tokens = tokenizer.encode(prompt, prepend=bos)

print(f"\n\n{'='*60}")
print("Test Generation")
print(f"{'='*60}")
print(f"Prompt: {prompt}\n")

# Normal generation
print("【Normal Generation】")
t0 = time.time()
normal_tokens = []
for token_col, mask_col in engine.generate(
    prompt_tokens, num_samples=1, max_tokens=100, temperature=0.8, seed=42
):
    token = token_col[0]
    normal_tokens.append(token)
    print(tokenizer.decode([token]), end="", flush=True)
t_normal = time.time() - t0
print(f"\nTime: {t_normal:.3f}s, Speed: {len(normal_tokens)/t_normal:.2f} tokens/s\n")

# Speculative generation
print("【Speculative Generation】")
t0 = time.time()
spec_tokens = []
final_stats = {}
for token_col, mask_col, stats in engine.generate_speculative(
    prompt_tokens, draft_model, num_samples=1, 
    max_tokens=100, temperature=0.8, seed=42, gamma=4
):
    token = token_col[0]
    spec_tokens.append(token)
    final_stats = stats
    print(tokenizer.decode([token]), end="", flush=True)
t_spec = time.time() - t0
print(f"\nTime: {t_spec:.3f}s, Speed: {len(spec_tokens)/t_spec:.2f} tokens/s")

# Statistics
acceptance_rate = final_stats["total_accepted"] / final_stats["total_drafted"] \
    if final_stats["total_drafted"] > 0 else 0
speedup = t_normal / t_spec

print(f"\n{'='*60}")
print("Statistics")
print(f"{'='*60}")
print(f"Speedup: {speedup:.2f}x")
print(f"Acceptance rate: {acceptance_rate:.1%}")
print(f"Average accepted per iteration: {final_stats['total_accepted']/final_stats['iterations']:.2f} tokens")
print(f"Total iterations: {final_stats['iterations']}")