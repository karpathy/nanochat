"""
Quick sanity check: verify Engine.generate produces identical output to model.generate.

Run from the project root:
    uv run python dev/test_engine_vs_generate.py
"""

import time

import torch

from nanochat.common import autodetect_device_type, compute_init
from nanochat.evaluation.engine import Engine
from nanochat.training.checkpoint import load_model

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

model, tokenizer, meta = load_model("base", device, phase="eval")
bos_token_id = tokenizer.get_bos_token_id()
kwargs = dict(max_tokens=64, temperature=0.0)
prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)

# Reference: model.generate
generated_tokens = []
torch.cuda.synchronize()
t0 = time.time()
for token in model.generate(prompt_tokens, **kwargs):
    generated_tokens.append(token)
    print(tokenizer.decode([token]), end="", flush=True)
print()
torch.cuda.synchronize()
print(f"Reference time: {time.time() - t0:.2f}s")
reference_ids = generated_tokens

# Engine.generate
generated_tokens = []
engine = Engine(model, tokenizer)
torch.cuda.synchronize()
t0 = time.time()
for token_column, _ in engine.generate(prompt_tokens, num_samples=1, **kwargs):
    token = token_column[0]
    generated_tokens.append(token)
    print(tokenizer.decode([token]), end="", flush=True)
print()
torch.cuda.synchronize()
print(f"Engine time: {time.time() - t0:.2f}s")

for i in range(len(reference_ids)):
    if reference_ids[i] != generated_tokens[i]:
        print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
        break
print(f"Match: {reference_ids == generated_tokens}")
