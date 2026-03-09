"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
"""

import os
from collections import deque

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import argparse

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print0,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Midtrain a pretrained model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--step", type=int, default=None, help="step to load the model from")
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of steps (-1 = one epoch)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (Adam)")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay (Adam)")
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--dry-run", action="store_true", help="log to wandb but don't save checkpoints/report")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-mid", name=args.run, config=user_config)
)

# Load the model and tokenizer
model, tokenizer, meta = load_model(
    "base", device, phase="train", model_tag=args.model_tag, step=args.step
)
pretrain_batch_size = meta.get("device_batch_size", None)
device_batch_size = args.device_batch_size
max_seq_len = args.max_seq_len
total_batch_size = args.total_batch_size
num_iterations = args.num_iterations

if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(
        f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?"
    )
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers
# Override the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# Midtraining data mixture and DataLoader
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture(
    [
        SmolTalk(split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
    ]
)
val_dataset = TaskMixture(
    [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ]
)

last_step = False
approx_progress = 0.0


def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1
    token_buffer = deque()
    scratch = torch.empty(
        needed_tokens, dtype=torch.int64, pin_memory=(device_type == "cuda")
    )
    cursor = ddp_rank
    it = 0
    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train":
                    last_step = True
        it += 1
        if num_iterations > 0 and it >= num_iterations:
            last_step = True
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        if split == "train":
            if num_iterations > 0:
                approx_progress = it / num_iterations
            else:
                approx_progress = cursor / dataset_size
        yield inputs, targets


train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0

eval_every = args.eval_every
eval_tokens = args.eval_tokens


# Learning rate scheduler
def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Synchronize last_step across all ranks
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb
    if eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps_count = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps_count, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # save checkpoint at the end
    if master_process and last_step and not args.dry_run:
        output_dirname = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config,
            },
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        logits, loss, combined_aux_loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
        progress = max(progress, approx_progress)
    # step the optimizers
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = gpu_peak_flops * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec
    if step > 10:
        total_training_time += dt
    print0(
        f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 10 == 0:
        log_dict = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if combined_aux_loss is not None:
            log_dict["train/ce_loss"] = combined_aux_loss["ce_loss"].item()
            log_dict["train/router_z_loss"] = combined_aux_loss["router_z_loss"].item()
            log_dict["train/load_balance_loss"] = combined_aux_loss["load_balance_loss"].item()
            if "compute_loss" in combined_aux_loss:
                log_dict["train/compute_loss"] = combined_aux_loss["compute_loss"].item()
            if "router_logits_abs_max" in combined_aux_loss:
                log_dict["train/router_logits_abs_max"] = combined_aux_loss["router_logits_abs_max"].item()
                log_dict["train/router_logits_abs_mean"] = combined_aux_loss["router_logits_abs_mean"].item()
            if "expert_bias_abs_max" in combined_aux_loss:
                log_dict["train/expert_bias_abs_max"] = combined_aux_loss["expert_bias_abs_max"].item()
                log_dict["train/expert_bias_abs_mean"] = combined_aux_loss["expert_bias_abs_mean"].item()
            if step % 100 == 0 and "expert_bias_per_layer" in combined_aux_loss:
                for i, bias_vec in enumerate(combined_aux_loss["expert_bias_per_layer"]):
                    log_dict[f"train/expert_bias_layer_{i}"] = wandb.Histogram(
                        bias_vec.float().cpu().numpy()
                    )
        wandb_run.log(log_dict)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanochat.report import get_report

    get_report().log(
        section="Midtraining",
        data=[
            user_config,
            {
                "Number of iterations": step,
                "DDP world size": ddp_world_size,
            },
            {
                "Minimum validation bpb": min_val_bpb,
            },
        ],
    )

# cleanup
wandb_run.finish()
compute_cleanup()
