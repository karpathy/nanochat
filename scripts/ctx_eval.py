"""
Multi-length BPB evaluation for context window extension experiments.

Evaluates a model's bits-per-byte at multiple sequence lengths to measure
how well it leverages increasing context. Usage:

    python -m scripts.ctx_eval --model-tag=picochat-ctx-s1 --step=5000 \
        --seq-lens=128,256,512,1024,2048 --device-batch-size=2

    python -m scripts.ctx_eval --model-tag=picochat-ctx-baseline-512 \
        --seq-lens=128,256,512,1024,2048 --device-batch-size=2
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch
import wandb

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, _patch_missing_config_keys, _patch_missing_keys
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Multi-length BPB evaluation")
parser.add_argument("--model-tag", type=str, required=True, help="model tag identifying the checkpoint directory")
parser.add_argument("--step", type=int, default=None, help="checkpoint step to load (default = last)")
parser.add_argument("--seq-lens", type=str, default="128,256,512,1024,2048", help="comma-separated sequence lengths to evaluate")
parser.add_argument("--device-batch-size", type=int, default=2, help="per-device batch size")
parser.add_argument("--eval-tokens", type=int, default=524288, help="number of tokens to evaluate per sequence length")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=vars(args), entity=os.environ.get("WANDB_ENTITY"))

# Parse sequence lengths
seq_lens = [int(s) for s in args.seq_lens.split(",")]

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)

# -----------------------------------------------------------------------------
# Load checkpoint

base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.model_tag)
step = args.step if args.step is not None else find_last_step(checkpoint_dir)
print0(f"Loading checkpoint: {checkpoint_dir} step {step}")

model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
if device.type in {"cpu", "mps"}:
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

model_config_kwargs = meta_data["model_config"]
_patch_missing_config_keys(model_config_kwargs)

# -----------------------------------------------------------------------------
# Evaluate at each sequence length

results = {}
print0(f"\n{'='*60}")
print0(f"Multi-Length BPB Evaluation — {args.model_tag} step {step}")
print0(f"{'='*60}")

for seq_len in seq_lens:
    print0(f"\nEvaluating at seq_len={seq_len}...")

    # Build model with overridden sequence_len
    config_kwargs = dict(model_config_kwargs)
    config_kwargs["sequence_len"] = seq_len
    model_config = GPTConfig(**config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()  # computes RoPE embeddings for the new seq_len
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Build val dataloader at this seq_len
    tokens_per_step = args.device_batch_size * seq_len * ddp_world_size
    eval_steps = max(1, args.eval_tokens // tokens_per_step)
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, seq_len, split="val", device=device
    )

    with autocast_ctx:
        bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)

    results[seq_len] = bpb
    print0(f"  seq_len={seq_len:5d} | val_bpb={bpb:.6f}")

    wandb_run.log({"seq_len": seq_len, "val_bpb": bpb})

    del model  # free memory before next iteration

# -----------------------------------------------------------------------------
# Summary

print0(f"\n{'='*60}")
print0(f"Summary: {args.model_tag} step {step}")
print0(f"{'='*60}")
print0(f"{'seq_len':>10s} | {'val_bpb':>10s}")
print0(f"{'-'*10}-+-{'-'*10}")
for seq_len in seq_lens:
    print0(f"{seq_len:>10d} | {results[seq_len]:>10.6f}")

# Save JSON results
if master_process:
    output_dir = os.path.join(base_dir, "ctx_eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_tag}_step{step}.json")
    with open(output_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "step": step, "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print0(f"\nResults saved to: {output_path}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Context extension evaluation", data=[
    {"model_tag": args.model_tag, "step": step},
    {f"bpb@{sl}": results[sl] for sl in seq_lens},
])

wandb_run.finish()
compute_cleanup()
