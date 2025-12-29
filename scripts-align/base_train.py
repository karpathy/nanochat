"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:

python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_iters=10 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import wandb

# Import from nanoMoE model (keeping train.py's original model)
import sys
sys.path.insert(0, '/root/nanoMoE')
from model import GPTConfig, GPT

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Data loading
use_bin_data = True # if True, use .bin files (nanoMoE format) instead of parquet/text
data_dir = "" # directory containing train.bin and val.bin files (only used if use_bin_data=True)
# Model architecture
depth = 6 # the depth of the Transformer model to train (matches nanoMoE n_layer=6), rest of the kwargs are derived
max_seq_len = 1024 # max context length (matches nanoMoE block_size=1024)
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+ (matches nanoMoE)
bias = False # do we use bias inside LayerNorm and Linear layers? (matches nanoMoE)
# MoE settings (matching nanoMoE config/train_nano_moe.py)
n_exp = 8 # number of experts (matches train_nano_moe.py)
top_k = 2 # number of active experts (matches train_nano_moe.py)
use_aux_loss = True # apply auxiliary loss (from Switch Transformer) (matches train_nano_moe.py)
use_router_z_loss = True # apply router z loss (from ST-MoE) (matches train_nano_moe.py)
use_noisy_top_k = False # use noisy top-k routing (matches train_nano_moe.py)
aux_loss_weight = 0.01 # auxiliary loss weight (matches train_nano_moe.py)
router_z_loss_weight = 0.001 # router z loss weight (matches train_nano_moe.py)
train_capacity = 1.25 # training capacity factor (matches train_nano_moe.py)
eval_capacity = 2.0 # evaluation capacity factor (matches train_nano_moe.py)
min_capacity = 4 # minimum batch size per expert (default from ST-MoE)
stride = 2 # one in every stride layers uses MoE (matches train_nano_moe.py)
use_switch_tfm_init = True # use weight init scheme from Switch Transformer (matches train_nano_moe.py)
switch_tfm_init_scale = 1.0 # scale for switch transformer init (matches train_nano_moe.py)
router_use_full_prec = True # use float32 in router (matches train_nano_moe.py)
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = 50000 # explicit number of steps (matches nanoMoE max_iters=50000, makes total tokens ~25B)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = -1 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 12 # per-device batch size (matches nanoMoE batch_size=12)
total_batch_size = 491520 # total desired batch size in #tokens (matches nanoMoE: 12 * 1024 * 40 = 491,520 for 8 GPUs)
embedding_lr = 0.0006 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.0006 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.1 # weight decay (matches nanoMoE weight_decay=1e-1)
matrix_lr = 0.0006 # learning rate for the matrix parameters (Muon)
learning_rate = 6e-4 # learning rate for AdamW optimizer (matches nanoMoE: 6e-4)
betas = (0.9, 0.95) # betas for AdamW optimizer (matches nanoMoE: beta1=0.9, beta2=0.95)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
decay_lr = True # whether to decay the learning rate (matches train_nano_moe.py)
# Learning rate decay parameters (matching train.py and train_nano_moe.py)
warmup_iters = 2000 # how many steps to warm up for (matches train.py default)
lr_decay_iters = 50000 # learning rate decay iterations (matches train_nano_moe.py)
min_lr = 6e-5 # minimum learning rate (matches train.py default, which equals 6e-4 * 0.1)
final_lr_frac = 0.1 # final learning rate as fraction of initial learning rate (for compatibility)

resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 500 # every how many steps to evaluate the model for val bpb (matches nanoMoE eval_interval=500)
eval_iters = 200 # number of iterations to evaluate val loss on (matches nanoMoE eval_iters=200)
log_interval = 10 # every how many steps to log training metrics (matches nanoMoE log_interval=10)
core_metric_every = -1 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = -1 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
save_every = 1000 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# System
compile = True # use PyTorch 2.0 to compile the model to be faster (matches nanoMoE)
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# Set random seed (matching nanoMoE/train.py)
seed_offset = ddp_rank if ddp else 0  # each process gets a different seed in DDP mode
torch.manual_seed(1337 + seed_offset)

# Set tf32 precision (matching nanoMoE/train.py)
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
# Use tiktoken GPT-2 tokenizer for compatibility with nanoMoE .bin data format
tokenizer = get_tokenizer(use_tiktoken_gpt2=True)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")
print0("Using tiktoken GPT-2 tokenizer (nanoMoE compatible)")

# Model kwargs are derived from the desired depth of the model
# For nanoMoE, we use n_layer, n_head, n_embd directly
n_layer = 6
model_dim = 384  # matches train_nano_moe.py
num_heads = 6  # matches train_nano_moe.py
n_head = num_heads
n_embd = model_dim
num_kv_heads = num_heads
print0(f"num_layers: {n_layer}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights (using nanoMoE GPT)
if not data_dir:
    # Default to nanoMoE data directory structure
    data_dir = "/root/nanoMoE/data/openwebtext"

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print0(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_config_kwargs = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd,
    block_size=max_seq_len, 
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304, 
    dropout=dropout, 
    bias=bias,
    # MoE parameters (matching train_nano_moe.py)
    n_exp=n_exp,
    top_k=top_k,
    use_aux_loss=use_aux_loss,
    use_router_z_loss=use_router_z_loss,
    use_noisy_top_k=use_noisy_top_k,
    aux_loss_weight=aux_loss_weight,
    router_z_loss_weight=router_z_loss_weight,
    train_capacity=train_capacity,
    eval_capacity=eval_capacity,
    min_capacity=min_capacity,
    stride=stride,
    use_switch_tfm_init=use_switch_tfm_init,
    switch_tfm_init_scale=switch_tfm_init_scale,
    router_use_full_prec=router_use_full_prec,
)

gptconf = GPTConfig(**model_config_kwargs)
model = GPT(gptconf)
model.to(device)

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d6
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
# if resuming:
#     print0(f"Resuming optimization from step {resume_from_step}")
#     model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
#     model.load_state_dict(model_data, strict=True, assign=True)
#     del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# Calculate FLOPs per token manually (based on PaLM paper Appendix B) before compilation
nparams_embedding = orig_model.transformer.wte.weight.numel()
num_params = sum(p.numel() for p in orig_model.parameters())
l, h, q, t = model_config_kwargs['n_layer'], model_config_kwargs['n_head'], model_config_kwargs['n_embd'] // model_config_kwargs['n_head'], model_config_kwargs['block_size']
num_flops_per_token = 6 * (num_params - nparams_embedding) + 12 * l * h * q * t
print0(f"Number of parameters: {num_params:,}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Initialize GradScaler (matching nanoMoE train.py - before optimizer)
# note: float16 data type will automatically use a GradScaler
dtype_actual = 'bfloat16' if device_type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
scaler = torch.cuda.amp.GradScaler(enabled=(dtype_actual == 'float16'))

# Initialize the Optimizer (AdamW for all parameters) - BEFORE DDP wrapping (matching nanoMoE)
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas, device_type=device_type)
adamw_optimizer = optimizer

# Compile the model (matching nanoMoE)
if compile:
    if master_process:
        print0("compiling the model... (takes a ~minute)")
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# Wrap model into DDP container (matching nanoMoE train.py)
from torch.nn.parallel import DistributedDataParallel as DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == "cuda" else None)

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# if resuming:
#     for opt, dat in zip(optimizer, optimizer_data):
#         if opt is not None and dat is not None:
#             opt.load_state_dict(dat)
#     del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Data loading (nanoMoE style - simple get_batch function)
if not data_dir:
    # Default to nanoMoE data directory structure
    data_dir = "/root/nanoMoE/data/openwebtext"
print0(f"Using .bin data loader from: {data_dir}")

# poor man's data loader (matching nanoMoE/train.py)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - max_seq_len, (device_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches (matching nanoMoE/train.py)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast_ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# fetch the very first batch
x, y = get_batch('train')

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler (cosine decay with warmup) - matching nanoMoE/train.py exactly
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    val_bpb = None  # Will be set during evaluation
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    val_bpb = None  # Will be set during evaluation

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # determine and set the learning rate for this iteration (matching nanoMoE/train.py)
    lr = get_lr(step) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # once in a while: evaluate the val loss (all ranks participate, matching nanoMoE/train.py)
    if last_step or step % eval_every == 0:
        losses = estimate_loss()
        val_loss = losses['val'].item()
        train_loss_eval = losses['train'].item()
        print0(f"Step {step:05d} | Train loss: {train_loss_eval:.4f}, Val loss: {val_loss:.4f}")
        if val_loss < min_val_bpb:
            min_val_bpb = val_loss
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/loss": val_loss,
            "train/loss_eval": train_loss_eval,
        })
        val_bpb = val_loss  # for compatibility with existing code

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer states
            { # metadata saved as json
                "step": step,
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16 (matching nanoMoE train.py exactly)
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with autocast_ctx:
            _, loss = model(x, y)  # nanoMoE model returns (logits, loss)
            loss = loss / grad_accum_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        x, y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    grad_clip_enabled = grad_clip > 0.0
    grad_norm = None
    if grad_clip_enabled:
        scaler.unscale_(optimizer)
        # clip_grad_norm_ returns the gradient norm before clipping
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    train_loss = loss.detach()  # for logging (after scaling)
    # -------------------------------------------------------------------------

    # logging (base_train.py style - keeping all the detailed logging)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    lossf = loss.item() * grad_accum_steps
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * lossf # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled and grad_norm is not None else ""
    lr_str = f"lr: {lr:.2e} |" if decay_lr else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} {lr_str}dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if decay_lr:
            log_data["lr"] = lr
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
