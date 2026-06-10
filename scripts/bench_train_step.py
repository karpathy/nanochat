"""
Micro-benchmark nanochat training step latency.

This script intentionally does not train a useful model. It runs a few synthetic
training steps and reports where time goes inside one optimizer step:

    forward -> backward -> optimizer.step -> zero_grad

Example:
python -m scripts.bench_train_step --depth=16 --device-batch-size=16 --steps=20 --warmup-steps=5

Distributed:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m scripts.bench_train_step --depth=16 --device-batch-size=16
"""

import argparse
import csv
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_peak_flops,
    is_ddp_initialized,
    print0,
)
from nanochat.gpt import GPT, GPTConfig


DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parents[1] / "docs" / "train_step_bench.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark nanochat training step latency")

    # Runtime
    parser.add_argument("--label", type=str, default="bench", help="experiment label written to stdout/CSV")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--steps", type=int, default=20, help="measured optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=5, help="unmeasured warmup optimizer steps")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="use torch.compile")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="CSV path for the one-line summary; pass an empty string to disable")

    # Model architecture, matching scripts.base_train defaults
    parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
    parser.add_argument("--n-kv-head", type=int, default=-1, help="number of key/value heads for GQA (-1 = match n_head)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="sequence length")
    parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
    parser.add_argument("--vocab-size", type=int, default=32768, help="synthetic token vocab size")

    # Optimization
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device micro-batch size")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="global tokens per optimizer step; -1 disables grad accumulation")
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
    parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters")
    parser.add_argument("--weight-decay", type=float, default=0.28, help="Muon weight decay")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon matrix learning rate")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="scalar learning rate")

    # FP8 training, matching scripts.base_train flags
    parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")

    return parser.parse_args()


def build_model(args, device):
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    num_kv_heads = num_heads if args.n_kv_head == -1 else args.n_kv_head
    assert num_heads % num_kv_heads == 0, f"n_head ({num_heads}) must be divisible by n_kv_head ({num_kv_heads})"

    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    print0(f"Model config: {asdict(config)}")
    model.to_empty(device=device)
    model.init_weights()
    return model


def maybe_enable_fp8(model, args, device_type):
    if not args.fp8:
        return
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA; ignoring --fp8")
        return

    import torch.nn as nn

    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        del fqn
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
    print0(f"FP8 enabled ({args.fp8_recipe}): converted {num_fp8}/{num_linear} linear layers")


def synchronize(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()


def elapsed_ms(fn, device_type):
    synchronize(device_type)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device_type)
    return out, (time.perf_counter() - t0) * 1000


def random_batch(args, device):
    x = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device)
    return x, y


def reduce_max(value, device):
    if not is_ddp_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def summarize(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    assert args.steps > 0, "--steps must be positive"
    assert args.warmup_steps >= 0, "--warmup-steps must be non-negative"
    assert args.device_batch_size > 0, "--device-batch-size must be positive"
    assert args.max_seq_len > 0, "--max-seq-len must be positive"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    del ddp, ddp_rank

    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_peak_flops = get_peak_flops(gpu_name)
        print0(f"GPU: {gpu_name} | Peak BF16 FLOPS/GPU: {gpu_peak_flops:.2e}")
    else:
        gpu_name = device_type
        gpu_peak_flops = float("inf")

    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    model = build_model(args, device)
    maybe_enable_fp8(model, args, device_type)

    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    flops_per_token = model.estimate_flops()
    print0(f"Parameters: {num_params:,}")
    print0(f"Estimated training FLOPs/token: {flops_per_token:.6e}")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
        scalar_lr=args.scalar_lr,
    )

    if args.compile:
        model = torch.compile(model, dynamic=False)

    micro_tokens_global = args.device_batch_size * args.max_seq_len * ddp_world_size
    total_batch_size = args.total_batch_size if args.total_batch_size > 0 else micro_tokens_global
    assert total_batch_size % micro_tokens_global == 0, (
        f"--total-batch-size ({total_batch_size}) must be divisible by "
        f"device_batch_size * max_seq_len * world_size ({micro_tokens_global})"
    )
    grad_accum_steps = total_batch_size // micro_tokens_global
    flops_per_step = flops_per_token * total_batch_size
    approx_forward_flops_per_step = flops_per_step / 3
    approx_backward_flops_per_step = flops_per_step * 2 / 3

    print0(f"World size: {ddp_world_size}")
    print0(f"Device batch size: {args.device_batch_size}")
    print0(f"Grad accumulation steps: {grad_accum_steps}")
    print0(f"Global tokens/optimizer step: {total_batch_size:,}")
    print0(f"Estimated training FLOPs/optimizer step: {flops_per_step:.6e}")

    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    metrics = {"forward_ms": [], "backward_ms": [], "optimizer_ms": [], "zero_grad_ms": [], "step_ms": []}
    total_steps = args.warmup_steps + args.steps

    for step in range(total_steps):
        measured = step >= args.warmup_steps
        step_times = {"forward_ms": 0.0, "backward_ms": 0.0}

        for _micro_step in range(grad_accum_steps):
            x, y = random_batch(args, device)

            loss, forward_ms = elapsed_ms(lambda: model(x, y), device_type)
            loss = loss / grad_accum_steps

            if scaler is not None:
                _, backward_ms = elapsed_ms(lambda: scaler.scale(loss).backward(), device_type)
            else:
                _, backward_ms = elapsed_ms(lambda: loss.backward(), device_type)

            step_times["forward_ms"] += forward_ms
            step_times["backward_ms"] += backward_ms

        def optimizer_step():
            if scaler is not None:
                scaler.unscale_(optimizer)
                if is_ddp_initialized():
                    for v in scaler._found_inf_per_device(optimizer).values():
                        dist.all_reduce(v, op=dist.ReduceOp.MAX)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        _, optimizer_ms = elapsed_ms(optimizer_step, device_type)
        _, zero_grad_ms = elapsed_ms(lambda: model.zero_grad(set_to_none=True), device_type)

        step_times["optimizer_ms"] = optimizer_ms
        step_times["zero_grad_ms"] = zero_grad_ms
        step_times["step_ms"] = sum(step_times.values())

        if measured:
            for key, value in step_times.items():
                metrics[key].append(reduce_max(value, device))

        if step == args.warmup_steps - 1:
            print0("Warmup complete; measuring...")

    summary = {key: summarize(values) for key, values in metrics.items()}
    mean_step_s = summary["step_ms"]["mean"] / 1000
    tok_per_sec = total_batch_size / mean_step_s
    flops_per_sec = flops_per_step / mean_step_s
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size) if device_type == "cuda" else 0.0

    row = {
        "label": args.label,
        "depth": args.depth,
        "world_size": ddp_world_size,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
        "grad_accum_steps": grad_accum_steps,
        "global_tokens_per_step": total_batch_size,
        "params": num_params,
        "flops_per_token": flops_per_token,
        "flops_per_step": flops_per_step,
        "approx_forward_flops_per_step": approx_forward_flops_per_step,
        "approx_backward_flops_per_step": approx_backward_flops_per_step,
        "forward_ms": summary["forward_ms"]["mean"],
        "backward_ms": summary["backward_ms"]["mean"],
        "optimizer_ms": summary["optimizer_ms"]["mean"],
        "zero_grad_ms": summary["zero_grad_ms"]["mean"],
        "step_ms": summary["step_ms"]["mean"],
        "forward_pct": 100 * summary["forward_ms"]["mean"] / summary["step_ms"]["mean"],
        "backward_pct": 100 * summary["backward_ms"]["mean"] / summary["step_ms"]["mean"],
        "optimizer_pct": 100 * summary["optimizer_ms"]["mean"] / summary["step_ms"]["mean"],
        "zero_grad_pct": 100 * summary["zero_grad_ms"]["mean"] / summary["step_ms"]["mean"],
        "tok_per_sec": tok_per_sec,
        "flops_per_sec": flops_per_sec,
        "approx_forward_flops_per_sec": approx_forward_flops_per_step / (summary["forward_ms"]["mean"] / 1000),
        "approx_backward_flops_per_sec": approx_backward_flops_per_step / (summary["backward_ms"]["mean"] / 1000),
        "mfu_pct": mfu,
        "compile": args.compile,
        "fp8": args.fp8,
        "gpu": gpu_name,
    }

    print0("")
    print0(f"Benchmark summary for {args.label} (mean over measured steps; max-reduced across ranks):")
    print0(f"  forward:   {row['forward_ms']:9.2f} ms ({row['forward_pct']:5.1f}%)")
    print0(f"  backward:  {row['backward_ms']:9.2f} ms ({row['backward_pct']:5.1f}%)")
    print0(f"  optimizer: {row['optimizer_ms']:9.2f} ms ({row['optimizer_pct']:5.1f}%)")
    print0(f"  zero_grad: {row['zero_grad_ms']:9.2f} ms ({row['zero_grad_pct']:5.1f}%)")
    print0(f"  step:      {row['step_ms']:9.2f} ms")
    print0(f"  tok/sec:   {row['tok_per_sec']:9,.0f}")
    print0(f"  TFLOP/s:   {row['flops_per_sec'] / 1e12:9.2f}")
    print0(f"  fwd TF/s:  {row['approx_forward_flops_per_sec'] / 1e12:9.2f} (approx)")
    print0(f"  bwd TF/s:  {row['approx_backward_flops_per_sec'] / 1e12:9.2f} (approx)")
    if device_type == "cuda":
        print0(f"  MFU:       {row['mfu_pct']:9.2f}%")

    if args.output_csv and (not is_ddp_initialized() or dist.get_rank() == 0):
        exists = os.path.exists(args.output_csv)
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print0(f"Wrote CSV summary to {args.output_csv}")

    compute_cleanup()


if __name__ == "__main__":
    main()
