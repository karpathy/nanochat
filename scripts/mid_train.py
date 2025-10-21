"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
"""

from collections import deque
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
import torch.distributed as dist

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.nemotron import Nemotron

# -----------------------------------------------------------------------------
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
dtype = "bfloat16"
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # initial learning rate is this fraction of the base learning rate
weight_decay = 0.0
eval_every = 150
eval_tokens = 20*524288
total_batch_size = 524288
dry_run = 0 # dry_run=1 is for experiments: we will log to wandb but we won't write checkpoints or report
dataset_choice = "smoltalk" # dataset choice: "smoltalk" or "nemotron"
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device_batch_size to this script?")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# Load tokenizer_name from checkpoint metadata
tokenizer_name = meta.get("tokenizer_name", "tokenizer")
print0(f"Using tokenizer: {tokenizer_name}")
token_bytes = get_token_bytes(tokenizer_name, device=device)

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers
# Override the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# Midtraining data mixture and DataLoader
base_dir = get_base_dir()

# Select dataset based on dataset_choice parameter
print0(f"Using dataset: {dataset_choice}")
if dataset_choice == "smoltalk":
    # Original: SmolTalk + MMLU + GSM8K
    train_dataset = TaskMixture([
        SmolTalk(split="train"), # 460K rows of general conversations
        MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems
        GSM8K(subset="main", split="train"), # 8K rows teaching simple math
    ]) # total: 460K + 100K + 8K = 568K rows
    val_dataset = TaskMixture([
        SmolTalk(split="test"), # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200), # 5.2K rows to match train ratios
        GSM8K(subset="main", split="test", stop=420), # 420 rows to match train ratios
    ]) # total: ~29.6K rows
elif dataset_choice == "nemotron":
    # Ablation: Nemotron with stem, math, chat, code (sampled to match SmolTalk 460K) + MMLU + GSM8K
    # Original Nemotron distribution: stem(355K/25.4%), math(239K/17.1%), chat(628K/44.9%), code(175K/12.5%)
    # Proportionally sampled to 460K total, then add MMLU + GSM8K to match SmolTalk structure
    train_dataset = TaskMixture([
        Nemotron(categories=["stem"], split="train", stop=151800),
        Nemotron(categories=["math"], split="train", stop=151800),
        Nemotron(categories=["chat"], split="train", stop=4600),
        Nemotron(categories=["code"], split="train", stop=151800),
        MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems
        GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
    ]) # total: 110K + 110K + 110K + 110K + 100K + 8K = 548K rows (similar to SmolTalk)
    # For validation, match SmolTalk validation set structure
    val_dataset = TaskMixture([
        Nemotron(categories=["stem"], split="train", start=151800, stop=155000),
        Nemotron(categories=["math"], split="train", start=151800, stop=155000),
        Nemotron(categories=["chat"], split="train", start=4600, stop=10000),
        Nemotron(categories=["code"], split="train", start=151800, stop=155000),
        MMLU(subset="all", split="test", stop=5200), # 5.2K rows to match train ratios
        GSM8K(subset="main", split="test", stop=420), # 420 rows to match train ratios
    ]) # total: 6.0K + 4.0K + 10.8K + 3.2K + 5.2K + 0.42K = 30.6K rows (similar to SmolTalk)
else:
    raise ValueError(f"Unknown dataset_choice: {dataset_choice}. Must be 'smoltalk' or 'nemotron'")
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1 # to form one training batch of inputs,targets
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    cursor = ddp_rank # increments by ddp_world_size each time, so each rank processes unique documents
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size # wrap around for another epoch
                if split == "train":
                    last_step = True # toggle last_step to True, which will terminate the training loop
        # Build up inputs/targets and yield
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        if split == "train":
            approx_progress = cursor / dataset_size # approximate progress as a fraction of the dataset
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate scheduler
def get_lr_multiplier(progress):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step and not dry_run:
        # Use model_tag from config if provided, otherwise default to d{depth}
        # output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", run)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config, # inputs to the training script,
                "tokenizer_name": tokenizer_name, # save tokenizer name for later loading
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically
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
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
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

# print a few more stats
print0(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not dry_run:
    from nanochat.report import get_report
    get_report(exp_name=run).log(section="Midtraining", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
