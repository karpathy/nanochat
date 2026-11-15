"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import argparse
import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import (
    apply_sft_resume_state,
    assert_resume_config_matches,
    find_safe_resume_step,
    load_checkpoint,
    load_model,
    load_optimizer_state,
    save_checkpoint,
    should_save_checkpoint,
)
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    check_finite_loss,
    compute_cleanup,
    compute_eta_str,
    compute_init,
    format_duration,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    maybe_torch_compile,
    print0,
)
from nanochat.defaults import DEFAULT_DEVICE_BATCH_SIZE, DEFAULT_SEQ_LEN
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.loss_eval import evaluate_bpb
from nanochat.memory import (
    auto_fit_echo_lines,
    auto_fit_enabled,
    empty_device_cache,
    fit_device_batch_size,
)
from nanochat.run_record import make_run_logger
from nanochat.term_style import paint
from nanochat.sft_loader import SftLoaderState
from nanochat.tokenizer import get_token_bytes
from scripts.chat_eval import run_chat_eval
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--auto-fit-batch", action=argparse.BooleanOptionalAction, default=None, help="shrink the device batch size until the estimated peak fits device memory (default: on for mps, off for cuda/cpu)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
# Resume / mid-training save
parser.add_argument("--save-every-secs", type=float, default=300.0, help="save checkpoint every N seconds during training (0 = disable)")
parser.add_argument("--save-every", type=int, default=0, help="save checkpoint every N steps (0 = disable)")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume from this step (-1 = auto-detect from checkpoint dir)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = make_run_logger(DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config), config=user_config)

# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Load the model and tokenizer
empty_device_cache(device) # hand the weights a clean allocator; MPS OOMs on the load itself otherwise
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       DEFAULT_SEQ_LEN,          meta),
    ("device_batch_size", DEFAULT_DEVICE_BATCH_SIZE, meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

# Memory preflight: run before the optimizer state and the SFT dataloader size themselves
# off the batch. There is no profile layer here - the batch came from the CLI or the checkpoint.
if auto_fit_enabled(args.auto_fit_batch, device_type):
    fitted_batch_size, fit_report = fit_device_batch_size(
        args.device_batch_size, depth=model.config.n_layer, seq_len=args.max_seq_len,
        device_type=device_type, vocab_size=tokenizer.get_vocab_size(),
    )
    batch_size_source = "CLI" if user_config["device_batch_size"] is not None else "checkpoint"
    for line in auto_fit_echo_lines(fit_report, [
        (batch_size_source, args.device_batch_size),
        ("fitted", fitted_batch_size),
    ]):
        print0(line)
    args.device_batch_size = fitted_batch_size

orig_model = model
# MPS: Inductor has no backend — maybe_torch_compile leaves the model eager (and echoes once).
model = maybe_torch_compile(model, device_type, dynamic=False)
depth = model.config.n_layer

base_dir = get_base_dir()
_sft_output_dirname = args.model_tag if args.model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", _sft_output_dirname)
_auto_detect_resume_step = args.resume_from_step
if _auto_detect_resume_step == -1:
    _auto_detect_resume_step = find_safe_resume_step(checkpoint_dir, ddp_world_size)
if _auto_detect_resume_step > 0:
    print0("=" * 60)
    print0(f"Resuming SFT from step {_auto_detect_resume_step}")
    print0("=" * 60)
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Checkpoint directory for mid-training saves (and auto-resume) is defined above with depth.

# Use pre-detected resume step (banner already printed above if resuming)
if args.resume_from_step == -1:
    args.resume_from_step = _auto_detect_resume_step

# Resume: read the SFT model + optimizer this rank saved, and the loader position.
# Config-mismatch guard: refuse to resume if key training dimensions changed.
_guard_keys = ["device_batch_size", "max_seq_len", "total_batch_size", "num_iterations", "world_size"]
if args.resume_from_step > 0:
    _sft_resume_model_data, _sft_resume_optim_data, _saved_meta = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    assert_resume_config_matches(
        _saved_meta.get("user_config", _saved_meta),
        {**{k: getattr(args, k) for k in _guard_keys[:-1]}, "world_size": ddp_world_size},
        _guard_keys, stage="SFT", hint="Use FORCE_STAGES=chat_sft to restart from scratch",
    )
    _sft_resume_loader_state = SftLoaderState.from_dict(
        _saved_meta.get("dataloader_state_dict") or {}, ddp_rank, ddp_world_size
    )
    _sft_resume_loop_state = _saved_meta.get("loop_state") or {}
else:
    _sft_resume_model_data = _sft_resume_optim_data = None
    _sft_resume_loader_state = SftLoaderState.initial(ddp_rank)
    _sft_resume_loop_state = {}

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
if args.resume_from_step > 0:
    # Resume restores the SFT weights and optimizer, not the base ones, so earlier SFT training is kept.
    apply_sft_resume_state(orig_model, optimizer, _sft_resume_model_data, _sft_resume_optim_data)
    del _sft_resume_model_data, _sft_resume_optim_data
    print0(f"Restored SFT model and optimizer state from step {args.resume_from_step}")
elif args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate.
# Skipped on resume: the restored optimizer already carries its scheduled LRs (no cumulative decay).
if args.resume_from_step > 0 and args.init_lr_frac != 1.0:
    print0(f"NOTE: --init-lr-frac={args.init_lr_frac} ignored on resume; "
           f"optimizer LRs are restored from the step {args.resume_from_step} checkpoint "
           f"(pass FORCE_STAGES=chat_sft to restart from step 0 with the new LR scale).")
if args.resume_from_step <= 0:
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
train_tasks = [
    SmolTalk(split="train"), # 460K rows of general conversations
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
]
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
def sft_data_generator_bos_bestfit(split, buffer_size=100, resume_state=None):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).

    For the train split, yields (inputs, targets, loader_state) triples where
    loader_state is an SftLoaderState snapshot taken BEFORE the batch is built, so
    resuming from that state replays the same batch.  Val yields (inputs, targets).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Initialise position from resume_state or from scratch
    if resume_state is not None and split == "train":
        s = resume_state
        cursor = s.cursor
        consumed = s.consumed
        epoch = s.epoch
        it = s.it
        # conv_buffer items are stored as [ids, mask] lists in JSON; keep as lists
        conv_buffer = [tuple(pair) for pair in s.conv_buffer]
    else:
        conv_buffer = []
        cursor = ddp_rank
        consumed = ddp_rank
        epoch = 1
        it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        # Snapshot BEFORE the batch so a resume replays the same batch.
        if split == "train":
            loader_state = SftLoaderState(
                cursor=cursor, consumed=consumed, epoch=epoch, it=it,
                conv_buffer=[[list(ids), list(mask)] for ids, mask in conv_buffer],
            )

        rows = []
        mask_rows = []
        row_lengths = []  # Track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # Row is now full (with padding)

            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Stopping condition to respect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        if split == "train":
            yield inputs, targets, loader_state
        else:
            yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train", resume_state=_sft_resume_loader_state)
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y, sft_loader_state = next(train_loader) # prefetch the very first batch of data
val_bpb = float("inf") # updated after each eval; inf until first eval runs
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = max(0, args.resume_from_step)
last_save_time = time.monotonic()

# Restore loop state when resuming
if _sft_resume_loop_state:
    min_val_bpb = _sft_resume_loop_state.get("min_val_bpb", float("inf"))
    smooth_train_loss = _sft_resume_loop_state.get("smooth_train_loss", 0)
    total_training_time = _sft_resume_loop_state.get("total_training_time", 0)
    progress = _sft_resume_loop_state.get("progress", 0)
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(paint("done", f"Step {step:05d} | Validation bpb: {val_bpb:.4f}"))
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(paint("done", f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}"))
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # mid-training time-based save + final save at last_step
    elapsed = time.monotonic() - last_save_time
    raw_time_due = should_save_checkpoint(step, args.resume_from_step, elapsed,
                                          args.save_every_secs, 0, False)
    if ddp and args.save_every_secs > 0:
        due_flag = torch.tensor([int(master_process and raw_time_due)], device=device)
        dist.broadcast(due_flag, src=0)
        raw_time_due = bool(due_flag.item())
    agreed_elapsed = float("inf") if raw_time_due else 0.0
    if should_save_checkpoint(step, args.resume_from_step, agreed_elapsed,
                              args.save_every_secs, args.save_every, last_step):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": args.total_batch_size,
                "num_iterations": args.num_iterations,
                "world_size": ddp_world_size,
                "dataloader_state_dict": sft_loader_state.to_dict(),
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "progress": progress,
                },
            },
            rank=ddp_rank,
        )
        last_save_time = time.monotonic()

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    if step == 0:
        print0(f"Step {step:05d} | training forward/backward ({grad_accum_steps} micro-steps)...")
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        check_finite_loss(train_loss, step, y)  # fail fast before EMA can absorb NaN
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, sft_loader_state = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # ETA: fail-soft — a failure must never abort training
    try:
        steps_done = step - 10
        eta_str = compute_eta_str(total_training_time, steps_done, args.num_iterations)
    except Exception:
        eta_str = ""
    total_str = f"/{args.num_iterations:05d}" if args.num_iterations > 0 else ""
    print0(f"step {step:05d}{total_str} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {format_duration(total_training_time)}{eta_str}")
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
            "train/epoch": current_epoch,
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {format_duration(total_training_time)}")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
if master_process:  # marker: last action of a successful run (shell SKIP gate; see runs/_stage_lib.sh)
    marker_dir = os.path.join(get_base_dir(), "chatsft_checkpoints")
    os.makedirs(marker_dir, exist_ok=True)
    open(os.path.join(marker_dir, ".train_ok"), "w").close()
    print0(f"Wrote completion marker: {os.path.join(marker_dir, '.train_ok')}")
