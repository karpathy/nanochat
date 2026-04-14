"""
Profile a single training step of nanochat (forward + backward + optimizer).
Outputs nsys and ncu reports for detailed GPU kernel analysis.

Usage:
    # Nsight Systems (full timeline):
    nsys profile -o profile_nsys_d6 python -m scripts.profile_step --depth 6
    nsys profile -o profile_nsys_d24 python -m scripts.profile_step --depth 24

    # NCU (kernel-level, split by phase to keep reports manageable):
    ncu --set full -o profile_ncu_d6_fwd  python -m scripts.profile_step --depth 6  --phase fwd
    ncu --set full -o profile_ncu_d6_bwd  python -m scripts.profile_step --depth 6  --phase bwd
    ncu --set full -o profile_ncu_d6_opt  python -m scripts.profile_step --depth 6  --phase opt
"""
import os
os.environ["NANOCHAT_BASE_DIR"] = os.path.expanduser("~/.cache/nanochat")

import argparse
import torch
import torch.cuda.nvtx as nvtx

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--phase", type=str, default="all", choices=["all", "fwd", "bwd", "opt"])
parser.add_argument("--seq-len", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--aspect-ratio", type=int, default=48)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
device = torch.device("cuda")
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Build model (same logic as base_train.py)
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.seq_len, vocab_size=32768,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern="SSSL",
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model = torch.compile(model, dynamic=False)
model.train()

optimizer = model.setup_optimizer(
    unembedding_lr=0.01, embedding_lr=0.01, scalar_lr=0.01,
    matrix_lr=0.01, weight_decay=0.1,
)

n_params = sum(p.numel() for p in model.parameters())
print0(f"Model: depth={args.depth} dim={model_dim} heads={num_heads} params={n_params:,}")
print0(f"Batch: {args.batch_size} x {args.seq_len} = {args.batch_size * args.seq_len:,} tokens")

# Dummy data
x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
y = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)

# ---------------------------------------------------------------------------
# Warmup (let torch.compile JIT)
print0("Warming up (torch.compile)...")
for _ in range(3):
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print0("Warmup done. Profiling...")

# ---------------------------------------------------------------------------
# Profiled step — NVTX ranges for nsys, CUDA ranges for ncu

def do_forward():
    nvtx.range_push("forward")
    loss = model(x, y)
    torch.cuda.synchronize()
    nvtx.range_pop()
    return loss

def do_backward(loss):
    nvtx.range_push("backward")
    loss.backward()
    torch.cuda.synchronize()
    nvtx.range_pop()

def do_optimizer():
    nvtx.range_push("optimizer")
    optimizer.step()
    torch.cuda.synchronize()
    nvtx.range_pop()
    model.zero_grad(set_to_none=True)

if args.phase == "fwd":
    torch.cuda.cudart().cudaProfilerStart()
    loss = do_forward()
    torch.cuda.cudart().cudaProfilerStop()
    print0(f"Forward done. loss={loss.item():.4f}")

elif args.phase == "bwd":
    loss = model(x, y)  # unprofiled forward
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    do_backward(loss)
    torch.cuda.cudart().cudaProfilerStop()
    print0("Backward done.")

elif args.phase == "opt":
    loss = model(x, y)  # unprofiled forward+backward
    loss.backward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    do_optimizer()
    torch.cuda.cudart().cudaProfilerStop()
    print0("Optimizer done.")

else:  # "all"
    torch.cuda.cudart().cudaProfilerStart()
    loss = do_forward()
    do_backward(loss)
    do_optimizer()
    torch.cuda.cudart().cudaProfilerStop()
    print0(f"Full step done. loss={loss.item():.4f}")

peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print0(f"Peak VRAM: {peak_mb:.0f} MiB")
