from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from dev.benchmark_mlx_reference import build_reference_batch, get_memory_stats, load_tokenizer_metadata
from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
from dev.mlx_gpt_prototype import MLXGPTPrototype, build_reference_config


def tree_leaves(tree) -> list[mx.array]:
    if isinstance(tree, dict):
        leaves = []
        for value in tree.values():
            leaves.extend(tree_leaves(value))
        return leaves
    if isinstance(tree, list):
        leaves = []
        for value in tree:
            leaves.extend(tree_leaves(value))
        return leaves
    if isinstance(tree, tuple):
        leaves = []
        for value in tree:
            leaves.extend(tree_leaves(value))
        return leaves
    return [tree] if isinstance(tree, mx.array) else []


def tree_l2_norm(tree) -> float:
    leaves = tree_leaves(tree)
    if not leaves:
        return 0.0
    total_sq = None
    for leaf in leaves:
        leaf_sq = mx.sum(mx.square(leaf.astype(mx.float32)))
        total_sq = leaf_sq if total_sq is None else total_sq + leaf_sq
    mx.eval(total_sq)
    return float(mx.sqrt(total_sq).item())


def tree_nonfinite_count(tree) -> int:
    leaves = tree_leaves(tree)
    if not leaves:
        return 0
    total = None
    for leaf in leaves:
        invalid = mx.sum(mx.logical_not(mx.isfinite(leaf)))
        total = invalid if total is None else total + invalid
    mx.eval(total)
    return int(total.item())


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = statistics.fmean(values)
    if mean_value == 0.0:
        return 0.0
    return statistics.pstdev(values) / mean_value


def init_model(model: MLXGPTPrototype, args) -> dict[str, object] | None:
    if args.init_from_pytorch_reference:
        return initialize_mlx_from_pytorch_reference(model)
    if args.pytorch_checkpoint_source is not None:
        return initialize_mlx_from_checkpoint_source(
            model,
            source=args.pytorch_checkpoint_source,
            model_tag=args.pytorch_model_tag,
            step=args.pytorch_step,
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short MLX training sanity check with explicit success criteria")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--init-from-pytorch-reference", action="store_true")
    parser.add_argument("--pytorch-checkpoint-source", type=str, choices=["base", "sft", "rl"], default=None)
    parser.add_argument("--pytorch-model-tag", type=str, default=None)
    parser.add_argument("--pytorch-step", type=int, default=None)
    args = parser.parse_args()

    shared_vocab_size, bos_token_id, shared_tokenizer_used = load_tokenizer_metadata(args.vocab_size)
    config = build_reference_config(
        depth=args.depth,
        sequence_len=args.max_seq_len,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        vocab_size=shared_vocab_size,
        window_pattern=args.window_pattern,
    )

    model = MLXGPTPrototype(config)
    init_metadata = init_model(model, args)
    optimizer = model.build_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    inputs, targets = build_reference_batch(args.device_batch_size, args.max_seq_len, config.vocab_size, bos_token_id)
    loss_and_grad = nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))

    for _ in range(max(args.warmup_steps, 0)):
        warm_loss, warm_grads = loss_and_grad(inputs, targets)
        optimizer.update(model, warm_grads)
        mx.eval(warm_loss, model.parameters(), *optimizer.state_trees())

    mx.reset_peak_memory()
    per_step = []
    for step_index in range(args.steps):
        start = time.perf_counter()
        loss, grads = loss_and_grad(inputs, targets)
        grad_l2 = tree_l2_norm(grads)
        grad_nonfinite = tree_nonfinite_count(grads)
        optimizer.update(model, grads)
        param_l2 = tree_l2_norm(model.parameters())
        param_nonfinite = tree_nonfinite_count(model.parameters())
        mx.eval(loss, model.parameters(), *optimizer.state_trees())
        elapsed = time.perf_counter() - start
        memory = get_memory_stats()
        per_step.append({
            "step": step_index + 1,
            "loss": float(loss.item()),
            "step_time_s": elapsed,
            "tokens_per_s": (args.device_batch_size * args.max_seq_len) / elapsed if elapsed > 0 else 0.0,
            "grad_l2": grad_l2,
            "grad_nonfinite": grad_nonfinite,
            "param_l2": param_l2,
            "param_nonfinite": param_nonfinite,
            "memory": memory,
        })

    losses = [row["loss"] for row in per_step]
    step_times = [row["step_time_s"] for row in per_step]
    steady_state_step_times = step_times[1:] if len(step_times) > 1 else step_times
    tokens_per_s = [row["tokens_per_s"] for row in per_step]
    active_memory = [row["memory"]["active_gb"] for row in per_step]

    success_criteria = {
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "all_gradients_finite": all(row["grad_nonfinite"] == 0 for row in per_step),
        "all_parameters_finite": all(row["param_nonfinite"] == 0 for row in per_step),
        "gradient_signal_present": all(row["grad_l2"] > 0.0 for row in per_step),
        "loss_improves": losses[-1] < losses[0] * 0.99,
        "throughput_positive": all(value > 0.0 for value in tokens_per_s),
        "timing_stable": coefficient_of_variation(steady_state_step_times) < 0.25,
    }

    summary = {
        "config": {
            "depth": config.n_layer,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": config.sequence_len,
            "model_dim": config.n_embd,
            "heads": config.n_head,
            "vocab_size": config.vocab_size,
            "params_total": model.num_params(),
        },
        "initialization": init_metadata,
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "metrics_to_log": [
            "loss",
            "step_time_s",
            "tokens_per_s",
            "grad_l2",
            "grad_nonfinite",
            "param_l2",
            "param_nonfinite",
            "memory.active_gb",
            "memory.peak_gb",
            "memory.cache_gb",
        ],
        "success_thresholds": {
            "loss_improves_by": "final_loss < initial_loss * 0.99",
            "timing_stable_cv_lt": 0.25,
            "timing_stable_scope": "measured steps after the first measured iteration",
            "all_gradients_finite": True,
            "all_parameters_finite": True,
            "gradient_signal_present": True,
            "throughput_positive": True,
        },
        "aggregates": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "loss_drop_pct": ((losses[0] - losses[-1]) / losses[0]) * 100.0 if losses[0] != 0 else 0.0,
            "mean_step_time_s": statistics.fmean(step_times),
            "step_time_cv": coefficient_of_variation(step_times),
            "steady_state_step_time_cv": coefficient_of_variation(steady_state_step_times),
            "mean_tokens_per_s": statistics.fmean(tokens_per_s),
            "mean_grad_l2": statistics.fmean([row["grad_l2"] for row in per_step]),
            "mean_param_l2": statistics.fmean([row["param_l2"] for row in per_step]),
            "active_memory_span_gb": max(active_memory) - min(active_memory),
            "peak_memory_gb": max(row["memory"]["peak_gb"] for row in per_step),
        },
        "success": {
            "passed": all(success_criteria.values()),
            "criteria": success_criteria,
        },
        "per_step": per_step,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()