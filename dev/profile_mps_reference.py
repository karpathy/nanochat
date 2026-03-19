"""Profile the frozen nanochat PyTorch + MPS reference workload.

This script is intentionally narrow:
- one Apple Silicon reference tier
- synthetic inputs matching the baseline benchmark shape
- explicit timing split for forward, backward, optimizer step, and zero_grad
- MPS memory snapshots plus rough tensor ownership estimates
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from nanochat.common import bytes_to_gb, get_mps_memory_stats
from nanochat.gpt import GPT, GPTConfig


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def sum_tensor_bytes(tensors) -> int:
    return sum(tensor_nbytes(tensor) for tensor in tensors if tensor is not None)

def synchronize() -> None:
    torch.mps.synchronize()


def clear_mps_state() -> None:
    gc.collect()
    torch.mps.empty_cache()
    synchronize()


def build_config(depth: int, seq_len: int, aspect_ratio: int, head_dim: int, vocab_size: int, window_pattern: str) -> GPTConfig:
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def phase_time(fn) -> tuple[float, object]:
    synchronize()
    start = time.perf_counter()
    value = fn()
    synchronize()
    return time.perf_counter() - start, value


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for value in state.values():
                if torch.is_tensor(value):
                    total += tensor_nbytes(value)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the nanochat MPS reference workload")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recommended-budget-frac", type=float, default=0.9)
    args = parser.parse_args()

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine.")

    torch.manual_seed(42)
    device = torch.device("mps")
    clear_mps_state()

    config = build_config(
        depth=args.depth,
        seq_len=args.max_seq_len,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        window_pattern=args.window_pattern,
    )

    model = GPT(config)
    model.to(device)
    model.init_weights()
    optimizer = model.setup_optimizer(
        unembedding_lr=args.lr,
        embedding_lr=args.lr,
        matrix_lr=args.lr,
        weight_decay=0.0,
    )
    inputs = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), dtype=torch.long, device=device)
    targets = inputs.clone()

    for _ in range(max(args.warmup_steps, 0)):
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()

    timings = []
    memory_snapshots = []
    final_loss = None
    last_grad_bytes = 0

    for step in range(args.profile_steps):
        step_record = {"step": step + 1}

        model.zero_grad(set_to_none=True)
        synchronize()
        before_forward = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)

        forward_time, loss = phase_time(lambda: model(inputs, targets))
        after_forward = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)

        backward_time, _ = phase_time(loss.backward)
        after_backward = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)
        last_grad_bytes = sum_tensor_bytes(param.grad for param in model.parameters())

        optimizer_time, _ = phase_time(optimizer.step)
        after_step = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)

        zero_grad_time, _ = phase_time(lambda: model.zero_grad(set_to_none=True))
        after_zero_grad = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)

        final_loss = float(loss.item())
        step_record.update({
            "forward_s": forward_time,
            "backward_s": backward_time,
            "optimizer_s": optimizer_time,
            "zero_grad_s": zero_grad_time,
            "total_s": forward_time + backward_time + optimizer_time + zero_grad_time,
        })
        timings.append(step_record)
        memory_snapshots.append({
            "step": step + 1,
            "before_forward": before_forward,
            "after_forward": after_forward,
            "after_backward": after_backward,
            "after_step": after_step,
            "after_zero_grad": after_zero_grad,
        })

    param_bytes = sum_tensor_bytes(model.parameters())
    grad_bytes = last_grad_bytes
    optimizer_bytes = optimizer_state_bytes(optimizer)

    last_snapshot = memory_snapshots[-1]
    activation_allocated_gb = max(
        0.0,
        last_snapshot["after_forward"]["allocated_gb"] - last_snapshot["before_forward"]["allocated_gb"],
    )
    backward_extra_allocated_gb = max(
        0.0,
        last_snapshot["after_backward"]["allocated_gb"] - last_snapshot["after_forward"]["allocated_gb"],
    )

    mean_forward = sum(row["forward_s"] for row in timings) / len(timings)
    mean_backward = sum(row["backward_s"] for row in timings) / len(timings)
    mean_optimizer = sum(row["optimizer_s"] for row in timings) / len(timings)
    mean_zero_grad = sum(row["zero_grad_s"] for row in timings) / len(timings)
    mean_total = sum(row["total_s"] for row in timings) / len(timings)
    tokens_per_step = args.device_batch_size * args.max_seq_len

    summary = {
        "config": {
            "depth": args.depth,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.max_seq_len,
            "window_pattern": args.window_pattern,
            "model_dim": config.n_embd,
            "heads": config.n_head,
            "params_total": model.num_scaling_params()["total"],
        },
        "timing": {
            "forward_s": mean_forward,
            "backward_s": mean_backward,
            "optimizer_s": mean_optimizer,
            "zero_grad_s": mean_zero_grad,
            "total_s": mean_total,
            "tokens_per_s": tokens_per_step / mean_total if mean_total > 0 else 0.0,
            "phase_share": {
                "forward": mean_forward / mean_total if mean_total else 0.0,
                "backward": mean_backward / mean_total if mean_total else 0.0,
                "optimizer": mean_optimizer / mean_total if mean_total else 0.0,
                "zero_grad": mean_zero_grad / mean_total if mean_total else 0.0,
            },
        },
        "memory": {
            "last_snapshot": last_snapshot,
            "budget": {
                "recommended_budget_frac": args.recommended_budget_frac,
                "budget_limit_gb": last_snapshot["after_step"]["budget_limit_gb"],
                "budget_headroom_gb": last_snapshot["after_step"]["budget_headroom_gb"],
                "driver_headroom_gb": last_snapshot["after_step"]["headroom_gb"],
                "exceeds_budget": any(snapshot[phase]["exceeds_budget"] for snapshot in memory_snapshots for phase in snapshot if phase != "step"),
            },
            "tensor_estimates_gb": {
                "parameters": bytes_to_gb(param_bytes),
                "gradients": bytes_to_gb(grad_bytes),
                "optimizer_state": bytes_to_gb(optimizer_bytes),
                "forward_activation_delta_allocated": activation_allocated_gb,
                "backward_extra_allocated": backward_extra_allocated_gb,
            },
        },
        "loss": final_loss,
        "per_step": timings,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()