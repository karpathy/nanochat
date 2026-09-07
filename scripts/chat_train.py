"""
Supervised finetuning: continue training a base model on conversations.

The data is the SFT split written by scripts/chat_prepare.py: general chat plus a
multiple-choice format primer, rendered and packed into rows with the loss mask folded
into the targets. So this is base_train's loop on different rows: one pass over them
at a fraction of the base learning rates, with a short warmup and a warmdown to zero
over the second half, no weight decay, a fresh optimizer. Every other hyperparameter
comes from the base checkpoint.

torchrun --standalone --nproc_per_node=8 -m scripts.chat_train -- --model-tag=d12
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import argparse

import wandb
import torch

from nanochat.common import print0, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON

from harness.runtime import compute_init, compute_cleanup, DummyWandb, autodetect_device_type, get_peak_flops
from harness.experiment import format_record, format_invocation
from nanochat.tokenizer import get_token_bytes
from harness.checkpoint import save_checkpoint, load_model, get_checkpoint_dir
from nanochat.dataloader import data_loader, data_loader_batches, split_info
from evals.bpb import evaluate_bpb

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised finetuning (SFT) of a base model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="base model to finetune, e.g. d12 (default: the largest)")
parser.add_argument("--model-step", type=int, default=None, help="base checkpoint step (default: the last)")
parser.add_argument("--num-iterations", type=int, default=-1, help="steps to train for (-1 = one pass over the SFT rows)")
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val bpb every N steps, and at the end")
parser.add_argument("--eval-tokens", type=int, default=10*524288, help="tokens per val bpb evaluation")
args = parser.parse_args()
user_config = vars(args).copy()
print0(format_invocation(args))

# The SFT schedule: a fraction of the base learning rates; a short linear warmup so the
# fresh optimizer accumulates its buffers before stepping at full size; constant; then
# a linear warmdown to zero over the second half of training
INIT_LR_FRAC = 0.8
WARMUP_FRAC = 0.05
WARMDOWN_FRAC = 0.5

# -----------------------------------------------------------------------------
# Compute init and wandb logging
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0)) if device_type == "cuda" else float("inf")
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# -----------------------------------------------------------------------------
# The base model, and the hyperparameters of the run that trained it
model, tokenizer, meta = load_model("base", device, model_tag=args.model_tag, step=args.model_step)
depth = model.config.n_layer
model_tag = args.model_tag if args.model_tag else f"d{depth}"
base_config = meta["user_config"]
max_seq_len = meta["max_seq_len"]
device_batch_size = meta["device_batch_size"]
total_batch_size = meta["total_batch_size"]
fwd = torch.compile(model.forward, dynamic=False)
num_flops_per_token = model.estimate_flops()
token_bytes = get_token_bytes(device=device)
world_tokens_per_fwdbwd = device_batch_size * max_seq_len * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}"
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Finetuning {model_tag}: batch {total_batch_size:,} tokens = {device_batch_size} x {max_seq_len} x {ddp_world_size} ranks x {grad_accum_steps} accumulation steps")

# -----------------------------------------------------------------------------
# The data: the SFT rows written by scripts/chat_prepare.py, one pass over them by default
row_len, train_rows, data_vocab_size, _ = split_info("sft_train")
assert max_seq_len <= row_len, f"the base model's max_seq_len {max_seq_len} exceeds the SFT row length {row_len}"
rows_per_step = total_batch_size // max_seq_len
num_iterations = args.num_iterations if args.num_iterations > 0 else train_rows // rows_per_step
print0(f"SFT data: {train_rows:,} rows x {row_len} tokens; {num_iterations:,} steps of {rows_per_step} rows")
train_loader = data_loader(device_batch_size, max_seq_len, "sft_train", device=device)
build_val_loader = lambda: data_loader_batches(device_batch_size, max_seq_len, "sft_val", device=device)
x, y, _ = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# The optimizer: the base run's learning rates scaled down, and no weight decay (the
# base run had decayed it to zero by its end)
optimizer = model.setup_optimizer(
    unembedding_lr=base_config["unembedding_lr"] * INIT_LR_FRAC,
    embedding_lr=base_config["embedding_lr"] * INIT_LR_FRAC,
    scalar_lr=base_config["scalar_lr"] * INIT_LR_FRAC,
    matrix_lr=base_config["matrix_lr"] * INIT_LR_FRAC,
    weight_decay=0.0,
)

def get_lr_multiplier(it):
    warmup_steps = round(WARMUP_FRAC * num_iterations)
    warmdown_start = num_iterations - round(WARMDOWN_FRAC * num_iterations)
    if it < warmup_steps:
        return (it + 1) / warmup_steps
    if it < warmdown_start:
        return 1.0
    progress = (it - warmdown_start) / (num_iterations - warmdown_start)
    lrm = 1.0 - progress
    return lrm

# Momentum scheduler for the Muon optimizer (ramps up over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of the training loss
ema_beta = 0.9
total_training_time = 0 # wall-clock time of training, after the warmup steps
while True:
    last_step = step == num_iterations # the loop runs num_iterations+1 times so we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb on the assistant tokens of held-out conversations (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        eval_steps = args.eval_tokens // world_tokens_per_fwdbwd
        val_bpb = evaluate_bpb(fwd, build_val_loader(), eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, val_bpb)
        print0(format_record("eval", step=step, val_bpb=round(val_bpb, 6)))
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "val/bpb": val_bpb})

    # at the end: save the model. Weights only, nothing resumes from an SFT checkpoint.
    if last_step:
        checkpoint_dir = get_checkpoint_dir(model_tag, "chat") # e.g. experiments/<name>/d12/chat
        meta_data = {
            "step": step,
            "val_bpb": val_bpb,
            "model_config": meta["model_config"],
            "user_config": user_config,
        }
        save_checkpoint(checkpoint_dir, step, model.state_dict(), None, meta_data, rank=ddp_rank)
        break

    # one training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = fwd(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize the loss here
        loss.backward()
        x, y, _ = next(train_loader) # prefetch the next batch while the GPU is busy
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
    optimizer.step()
    optimizer.zero_grad() # zeroes the flat grad tapes in place (do NOT sever p.grad views)
    synchronize()
    dt = time.time() - t0
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** step)
    tok_per_sec = int(total_batch_size / dt)
    mfu = 100 * num_flops_per_token * total_batch_size / dt / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps (compile, warmup)
    print0(f"step {step:05d}/{num_iterations:05d} | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently; manage it manually
    if step == 1:
        gc.collect() # collect the garbage from setup
        gc.freeze() # exclude everything surviving now from future scans
        gc.disable()

# -----------------------------------------------------------------------------
# The stage record (see harness/experiment.py): this is what downstream tooling consumes
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
summary = {
    "model_tag": model_tag,
    "depth": depth,
    "num_iterations": step,
    "tokens_trained": total_batch_size * step,
    "train_time_sec": round(total_training_time, 1),
    "peak_vram_mib": round(get_max_memory() / 1024 / 1024),
    "val_bpb": round(val_bpb, 6),
    "min_val_bpb": round(min_val_bpb, 6),
}
print0(format_record("summary", **summary))

wandb_run.finish()
compute_cleanup()
