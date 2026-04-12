"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval
from tasks.arc import ARC
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Chat SFT training")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--source", type=str, default="mid", help="base|mid — which checkpoint to load")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--step", type=int, default=None, help="step to load the model from")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--device-batch-size", type=int, default=4, help="max batch size to avoid OOM")
parser.add_argument("--num-epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--num-iterations", type=int, default=-1, help="override number of iterations (-1 = use num_epochs)")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="target examples per optimization step")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (Adam)")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR as fraction of base LR")
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval-steps", type=int, default=100, help="number of eval steps")
parser.add_argument("--eval-metrics-every", type=int, default=200, help="evaluate metrics every N steps")
parser.add_argument("--eval-metrics-max-problems", type=int, default=1024, help="max problems per metric eval")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(
        project="nanochat-sft", name=args.run, config=user_config, save_code=True
    )
)

# Load the model and tokenizer
model, tokenizer, meta = load_model(
    args.source, device, phase="train", model_tag=args.model_tag, step=args.step
)
orig_model = model
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Task data mixture
identity_conversations_filepath = os.path.join(
    get_base_dir(), "identity_conversations.jsonl"
)
train_ds = TaskMixture(
    [
        ARC(subset="ARC-Easy", split="train"),
        ARC(subset="ARC-Challenge", split="train"),
        GSM8K(subset="main", split="train"),
        SmolTalk(split="train", stop=10_000),
        CustomJSON(filepath=identity_conversations_filepath),
    ]
)
val_ds = SmolTalk(split="test")

# -----------------------------------------------------------------------------
# DataLoader
device_batch_size = args.device_batch_size


def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, : n - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, : n - 1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []


examples_per_step = device_batch_size * ddp_world_size
target_examples_per_step = args.target_examples_per_step
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, (
    "Target examples per step must be divisible by examples per step"
)
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

num_iterations = args.num_iterations
num_epochs = args.num_epochs
if num_iterations == -1:
    assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
def build_val_loader(): return sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Training loop

eval_every = args.eval_every
eval_steps = args.eval_steps
eval_metrics_every = args.eval_metrics_every
eval_metrics_max_problems = args.eval_metrics_max_problems


def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm


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
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad():
                logits, loss, combined_aux_loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val_loss": val_loss})
        model.train()

    # evaluate accuracy of multiple choice tasks
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad():
            metrics["mmlu_acc"] = run_chat_eval(
                "MMLU", model, tokenizer, engine,
                batch_size=device_batch_size * 2,
                max_problems=eval_metrics_max_problems,
            )
            metrics["arc_easy_acc"] = run_chat_eval(
                "ARC-Easy", model, tokenizer, engine,
                batch_size=device_batch_size * 2,
                max_problems=eval_metrics_max_problems,
            )
        metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({"step": step, **metrics})
        model.train()

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        logits, loss, combined_aux_loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

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
    print0(
        f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}"
    )
    log_dict = {
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    }
    if combined_aux_loss is not None:
        log_dict["train/ce_loss"] = combined_aux_loss["ce_loss"].item()
        log_dict["train/router_z_loss"] = combined_aux_loss["router_z_loss"].item()
        log_dict["train/load_balance_loss"] = combined_aux_loss["load_balance_loss"].item()
        if "compute_loss" in combined_aux_loss:
            log_dict["train/compute_loss"] = combined_aux_loss["compute_loss"].item()
    wandb_run.log(log_dict)
    step += 1

# Save the model at the end
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None,
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        },
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report  # noqa: E402

get_report().log(
    section="Chat SFT",
    data=[
        user_config,
        {
            "Training rows": len(train_ds),
            "Number of iterations": num_iterations,
            "Training loss": train_loss_item,
            "Validation loss": val_loss,
        },
    ],
)

# Cleanup
wandb_run.finish()
compute_cleanup()
