"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
import argparse

import torch

import wandb
from nanochat.checkpoint_manager import save_checkpoint
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
    print_banner,
)
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.engine import Engine
from nanochat.flash_attention import USE_FA3
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="number of transformer layers")
parser.add_argument("--model-dim", type=int, default=-1, help="model dimension (-1 = derive from depth as depth * 64)")
parser.add_argument("--num-heads", type=int, default=-1, help="number of attention heads (-1 = derive from model_dim)")
parser.add_argument("--num-kv-heads", type=int, default=-1, help="number of kv heads for GQA (-1 = same as num_heads)")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context")
# MoE routing
parser.add_argument("--expert-sizes", type=json.loads, default=[(64, 256)], help="JSON list of [count, width] tuples, e.g. '[[64,256]]'")
parser.add_argument("--num-active-experts", type=int, default=8, help="top-k experts per token")
parser.add_argument("--load-balance-loss-weight", type=float, default=0.08, help="load balance loss weight")
parser.add_argument("--router-z-loss-weight", type=float, default=0.001, help="router z-loss weight")
parser.add_argument("--compute-loss-weight", type=float, default=0.004, help="compute loss weight")
parser.add_argument("--use-bias-balancing", action="store_true", help="enable bias balancing for expert routing")
parser.add_argument("--bias-update-speed", type=float, default=0.0005, help="SMEBU learning rate (lambda)")
parser.add_argument("--bias-momentum", type=float, default=0.5, help="SMEBU momentum factor (beta)")
parser.add_argument("--bias-kappa", type=float, default=2.0, help="SMEBU tanh saturation speed (kappa)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=20, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size (reduce if OOM)")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default="", help="override model tag for checkpoint directory name")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoint every N steps (-1 = only at end)")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# Convert expert_sizes from list of lists to list of tuples (JSON gives lists)
args.expert_sizes = [tuple(x) for x in args.expert_sizes]
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
if USE_FA3:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("Using PyTorch SDPA fallback (FA3 not available)")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs - use provided values or derive from depth
depth = args.depth
num_layers = depth
model_dim = args.model_dim
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
max_seq_len = args.max_seq_len
if model_dim == -1:
    model_dim = depth * 64  # default aspect ratio 64
if num_heads == -1:
    num_heads = max(1, (model_dim + 127) // 128)  # default head dim 128
if num_kv_heads == -1:
    num_kv_heads = num_heads  # default 1:1 GQA ratio (i.e. no GQA)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
device_batch_size = args.device_batch_size
total_batch_size = args.total_batch_size
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
    expert_sizes=args.expert_sizes,
    num_active_experts=args.num_active_experts,
    load_balance_loss_weight=args.load_balance_loss_weight,
    router_z_loss_weight=args.router_z_loss_weight,
    compute_loss_weight=args.compute_loss_weight,
    use_bias_balancing=args.use_bias_balancing,
    bias_update_speed=args.bias_update_speed,
    bias_momentum=args.bias_momentum,
    bias_kappa=args.bias_kappa,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model  # original, uncompiled model, for saving raw model state_dict
model = torch.compile(model, dynamic=False)
total_params = sum(p.numel() for p in model.parameters())
if model_config.use_moe:
    total_expert_width = sum(count * size for count, size in model_config.expert_sizes)
    num_experts = sum(count for count, _ in model_config.expert_sizes)
    moe_params = model_dim * total_expert_width * 2 * num_layers
    inactive_moe = (
        moe_params * (num_experts - model_config.num_active_experts) // num_experts
    )
    active_params = total_params - inactive_moe
    print0(f"Number of total parameters: {total_params:,}")
    print0(f"Number of active parameters: {active_params:,}")
else:
    print0(f"Number of parameters: {total_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")
num_params = active_params if model_config.use_moe else total_params

# Calculate number of iterations
num_iterations = args.num_iterations
target_flops = args.target_flops
target_param_data_ratio = args.target_param_data_ratio
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

# Initialize the DataLoaders for train/val
base_dir = get_base_dir()
train_loader = tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="train", device=device
)
def build_val_loader(): return tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="val", device=device
)
x, y = next(train_loader)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

warmup_ratio = args.warmup_ratio
warmdown_ratio = args.warmdown_ratio
final_lr_frac = args.final_lr_frac

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
save_every = args.save_every
eval_every = args.eval_every
eval_tokens = args.eval_tokens
core_metric_every = args.core_metric_every
core_metric_max_per_task = args.core_metric_max_per_task
sample_every = args.sample_every
model_tag = args.model_tag

# note that we run +1 steps only so that we can eval and save at the end
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
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

    # once in a while: estimate the CORE metric (all ranks participate)
    results = {}
    if core_metric_every > 0 and (
        last_step or (step > 0 and step % core_metric_every == 0)
    ):
        model.eval()
        results = evaluate_model(
            orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
        )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
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
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=16, temperature=0
            )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint periodically and at the end (only on master process)
    should_save = last_step or (save_every > 0 and step > 0 and step % save_every == 0)
    if master_process and should_save:
        output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
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
    # step the optimizers
    lrm = get_lr_multiplier(step)
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

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = gpu_peak_flops * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 100 == 0:
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
            if "expert_usage" in combined_aux_loss:
                for j, val in enumerate(combined_aux_loss["expert_usage"]):
                    log_dict[f"train/expert_usage_{j}"] = val.item()
            if step % 1000 == 0 and "expert_bias_per_layer" in combined_aux_loss:
                for i, bias_vec in enumerate(combined_aux_loss["expert_bias_per_layer"]):
                    for j, val in enumerate(bias_vec):
                        log_dict[f"train/expert_bias_layer_{i}_expert_{j}"] = val.item()
            if step % 1000 == 0 and "expert_usage_per_layer" in combined_aux_loss:
                for i, usage_vec in enumerate(combined_aux_loss["expert_usage_per_layer"]):
                    for j, val in enumerate(usage_vec):
                        log_dict[f"train/expert_usage_layer_{i}_expert_{j}"] = val.item()
        wandb_run.log(log_dict)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report  # noqa: E402

get_report().log(
    section="Base model training",
    data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
        },
        {
            "Minimum validation bpb": min_val_bpb,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()
compute_cleanup()
