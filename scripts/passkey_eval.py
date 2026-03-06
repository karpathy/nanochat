"""
Passkey retrieval (needle-in-a-haystack) evaluation for context extension.

Tests whether a model can retrieve a specific number hidden in filler text,
directly measuring information retrieval across the context window. Usage:

    python -m scripts.passkey_eval --model-tag=picochat-ctx-s1 --step=5000 \
        --seq-lens=256,512,1024,2048 --num-trials=50

    python -m scripts.passkey_eval --model-tag=smoke-test --step=30 \
        --seq-lens=512 --num-trials=5
"""

import os
import json
import random
import argparse
from contextlib import nullcontext

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, _patch_missing_config_keys, _patch_missing_keys
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type, get_base_dir
from nanochat.tokenizer import get_tokenizer

# Pool of neutral filler sentences
FILLER_SENTENCES = [
    "The weather was pleasant that afternoon.",
    "Trees lined both sides of the quiet street.",
    "A small bird landed on the windowsill.",
    "The river flowed steadily through the valley.",
    "Clouds drifted slowly across the blue sky.",
    "The old clock on the wall ticked softly.",
    "Leaves rustled gently in the autumn breeze.",
    "The path wound through a meadow of wildflowers.",
    "Sunlight filtered through the tall pine trees.",
    "A gentle rain began to fall at dusk.",
    "The village square was empty in the morning.",
    "Smoke rose from the chimney of the cottage.",
    "The cat slept peacefully on the warm stone.",
    "Waves lapped against the wooden dock.",
    "The garden was full of blooming roses.",
    "A distant bell chimed the hour.",
    "The library was quiet and dimly lit.",
    "Frost covered the grass in the early morning.",
    "The mountain trail was steep but well-marked.",
    "Stars appeared one by one in the evening sky.",
]

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Passkey retrieval evaluation")
parser.add_argument("--model-tag", type=str, required=True, help="model tag identifying the checkpoint directory")
parser.add_argument("--step", type=int, default=None, help="checkpoint step to load (default = last)")
parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048", help="comma-separated sequence lengths to evaluate")
parser.add_argument("--num-trials", type=int, default=50, help="number of passkey trials per sequence length")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

import wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=vars(args), entity=os.environ.get("WANDB_ENTITY"))

seq_lens = [int(s) for s in args.seq_lens.split(",")]
tokenizer = get_tokenizer()
bos_id = tokenizer.get_bos_token_id()

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
# Passkey sequence construction

def build_passkey_sequence(passkey: str, seq_len: int, rng: random.Random) -> list[int]:
    """Build a token sequence: BOS + prefix + filler + suffix, trimmed/padded to seq_len."""
    prefix = f"The special number is {passkey}."
    suffix = f" The special number mentioned above is {passkey}."

    prefix_tokens = tokenizer.encode(prefix)
    suffix_tokens = tokenizer.encode(suffix)

    # We need: [BOS] + prefix_tokens + filler_tokens + suffix_tokens = seq_len
    filler_budget = seq_len - 1 - len(prefix_tokens) - len(suffix_tokens)
    if filler_budget <= 0:
        # Sequence too short for this passkey format; just use prefix + suffix
        tokens = [bos_id] + prefix_tokens + suffix_tokens
        return tokens[:seq_len]

    # Generate filler tokens by repeatedly sampling sentences
    filler_tokens = []
    while len(filler_tokens) < filler_budget:
        sentence = " " + rng.choice(FILLER_SENTENCES)
        filler_tokens.extend(tokenizer.encode(sentence))

    filler_tokens = filler_tokens[:filler_budget]

    tokens = [bos_id] + prefix_tokens + filler_tokens + suffix_tokens
    # Trim to exact seq_len (suffix may cause slight overshoot due to tokenization)
    return tokens[:seq_len]


def find_passkey_positions(tokens: list[int], passkey: str) -> tuple[list[int], int]:
    """Find the token positions of the final passkey occurrence.

    Returns (passkey_token_ids, start_position) where start_position is
    the index in `tokens` where the final passkey begins.
    """
    passkey_token_ids = tokenizer.encode(passkey)
    # Search backwards for the last occurrence
    for i in range(len(tokens) - len(passkey_token_ids), -1, -1):
        if tokens[i:i + len(passkey_token_ids)] == passkey_token_ids:
            return passkey_token_ids, i
    raise ValueError(f"Passkey '{passkey}' not found in token sequence")


# -----------------------------------------------------------------------------
# Evaluate at each sequence length

results = {}
print0(f"\n{'='*60}")
print0(f"Passkey Retrieval Evaluation — {args.model_tag} step {step}")
print0(f"{'='*60}")

for seq_len in seq_lens:
    print0(f"\nEvaluating at seq_len={seq_len} ({args.num_trials} trials)...")

    # Build model with overridden sequence_len
    config_kwargs = dict(model_config_kwargs)
    config_kwargs["sequence_len"] = seq_len
    model_config = GPTConfig(**config_kwargs)
    _patch_missing_keys(model_data, model_config)

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    correct = 0
    for trial in range(args.num_trials):
        rng = random.Random(trial * 1000 + seq_len)
        passkey = str(rng.randint(10000, 99999))

        tokens = build_passkey_sequence(passkey, seq_len, rng)
        passkey_token_ids, passkey_start = find_passkey_positions(tokens, passkey)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad(), autocast_ctx:
            logits = model(input_ids)

        # Check predictions at positions just before each passkey token
        # logits[0, pos] predicts token at pos+1, so logits[0, passkey_start-1]
        # predicts the first passkey token, etc.
        trial_correct = True
        for k, expected_id in enumerate(passkey_token_ids):
            pred_pos = passkey_start - 1 + k  # position whose next-token we check
            if pred_pos < 0 or pred_pos >= seq_len:
                trial_correct = False
                break
            predicted_id = logits[0, pred_pos].argmax().item()
            if predicted_id != expected_id:
                trial_correct = False
                break

        if trial_correct:
            correct += 1

    accuracy = correct / args.num_trials
    results[seq_len] = accuracy
    print0(f"  seq_len={seq_len:5d} | accuracy={accuracy:.4f} ({correct}/{args.num_trials})")

    wandb_run.log({"seq_len": seq_len, "passkey_accuracy": accuracy})

    del model

# -----------------------------------------------------------------------------
# Summary

print0(f"\n{'='*60}")
print0(f"Summary: {args.model_tag} step {step}")
print0(f"{'='*60}")
print0(f"{'seq_len':>10s} | {'accuracy':>10s}")
print0(f"{'-'*10}-+-{'-'*10}")
for seq_len in seq_lens:
    print0(f"{seq_len:>10d} | {results[seq_len]:>10.4f}")

# Save JSON results
if master_process:
    output_dir = os.path.join(base_dir, "passkey_eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_tag}_step{step}.json")
    with open(output_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "step": step, "num_trials": args.num_trials, "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print0(f"\nResults saved to: {output_path}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Passkey retrieval evaluation", data=[
    {"model_tag": args.model_tag, "step": step, "num_trials": args.num_trials},
    {f"accuracy@{sl}": results[sl] for sl in seq_lens},
])

wandb_run.finish()
compute_cleanup()
