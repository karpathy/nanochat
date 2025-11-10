"""
Train a chat model with SIM-CoT (Supervised Implicit Chain-of-Thought).

This script adds step-level supervision to improve reasoning on GSM8K.
Based on the paper "SIM-CoT: Supervised Implicit Chain-of-Thought" (ICLR 2025).

Run on one GPU:
    python -m scripts.chat_simcot

Run with torchrun for distributed training:
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_simcot

Key differences from chat_sft.py:
- Uses SIMCoTGSM8K task with step boundary annotations
- Applies step-level loss weighting to upweight reasoning steps
- Tracks per-step accuracy metrics
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.simcot_utils import compute_step_weights, compute_step_accuracy
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.simcot_gsm8k import SIMCoTGSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# SIM-CoT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from
model_tag = None # model tag to load the model from
step = None # step to load the model from
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# SIM-CoT specific hyperparameters
step_weight_multiplier = 2.0 # how much to upweight reasoning steps (1.0 = no upweighting, 2.0 = 2x weight)
track_step_accuracy = True # whether to compute per-step accuracy metrics
# evaluation and logging
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
eval_metrics_max_problems = 1024
# now allow CLI to override the settings via the configurator
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-simcot", name=run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(model, tokenizer) # will be used for inline model evaluation only

# -----------------------------------------------------------------------------
# Task data mixture - focusing on GSM8K with SIM-CoT annotations
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    SIMCoTGSM8K(subset="main", split="train"), # 8K rows with step annotations
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
    CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    SimpleSpelling(size=300, split="train"), # 300 rows
    SpellingBee(size=300, split="train"), # 300 rows
]) # Total: ~23K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows

# -----------------------------------------------------------------------------
# DataLoader with step boundary tracking

def simcot_data_generator(dataset, batch_size):
    """
    Data generator that yields batches with step boundary information.
    """
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _, _ in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        step_boundaries_batch = []

        for i, (ids, mask, step_bounds) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]

            # Set targets with masking
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets

            # Adjust step boundaries (shift by 1 because of input/target offset)
            adjusted_bounds = [pos - 1 for pos in step_bounds if pos > 0 and pos < n]
            step_boundaries_batch.append(adjusted_bounds)

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets, step_boundaries_batch

    # Iterate over dataset
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)

            # Extract step boundaries if available
            step_boundaries = doc.get('step_boundaries', [])

            batch.append((ids, mask, step_boundaries))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step: {examples_per_step}")
print0(f"Step weight multiplier: {step_weight_multiplier}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = simcot_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: simcot_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# Set the initial learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Training loop with SIM-CoT step-level supervision

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets, val_step_bounds = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                # Use standard loss for validation
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # evaluate accuracy metrics
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            # Note: GSM8K evaluation can be added here if you have a chat_eval function for it
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # Training step with SIM-CoT step-level weighting
    num_tokens = torch.tensor(0, device=device)
    weighted_tokens = torch.tensor(0.0, device=device)
    step_acc_sum = 0.0
    step_acc_count = 0

    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets, step_boundaries = next(train_iter)

        with autocast_ctx:
            # Forward pass with per-token loss
            logits = model(train_inputs)  # (B, T, vocab_size)

            # Compute step-weighted loss
            import torch.nn.functional as F
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = train_targets.view(-1)

            # Per-token loss
            per_token_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=-1,
                reduction='none'
            ).view(B, T)

            # Compute step weights
            weights = compute_step_weights(train_targets, step_boundaries, step_weight_multiplier)

            # Weighted loss
            loss = (per_token_loss * weights).sum() / (weights.sum() + 1e-8)

            # Track step accuracy if enabled
            if track_step_accuracy:
                step_metrics = compute_step_accuracy(logits.detach(), train_targets, step_boundaries)
                step_acc_sum += step_metrics['step_accuracy']
                step_acc_count += 1

        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()

        num_tokens += (train_targets >= 0).sum()
        weighted_tokens += weights.sum()

    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(weighted_tokens, op=dist.ReduceOp.SUM)

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    weighted_tokens_item = weighted_tokens.item()
    avg_step_acc = step_acc_sum / step_acc_count if step_acc_count > 0 else 0.0

    log_str = f"Step {step:05d}/{num_iterations:05d} | Loss: {train_loss_item:.6f} | lrm: {lrm:.6f} | tokens: {num_tokens_item:,} | weighted: {weighted_tokens_item:.1f}"
    if track_step_accuracy:
        log_str += f" | step_acc: {avg_step_acc:.4f}"
    print0(log_str)

    log_dict = {
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
        "weighted_tokens": weighted_tokens_item,
    }
    if track_step_accuracy:
        log_dict["step_accuracy"] = avg_step_acc

    wandb_run.log(log_dict)
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsimcot_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # we don't save optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
            "step_weight_multiplier": step_weight_multiplier,
        }
    )
    print(f"✅ Saved SIM-CoT model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SIM-CoT Training", data=[
    user_config,
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
        "Step weight multiplier": step_weight_multiplier,
        "Final step accuracy": avg_step_acc if track_step_accuracy else "N/A",
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
