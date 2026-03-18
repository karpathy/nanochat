from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from dev.benchmark_mlx_reference import get_memory_stats, load_tokenizer_metadata
from dev.mlx_compile_utils import build_loss_and_grad, eval_training_state, make_training_step
from dev.mlx_input_batches import make_input_batch_provider
from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
from dev.mlx_gpt_prototype import MLXGPTPrototype, build_reference_config
from dev.mlx_logging import add_logging_args, write_summary_log


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
    parser.add_argument("--matrix-optimizer", type=str, choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-orthogonalize-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--muon-block-groups", type=str, choices=["all", "mlp_only", "attn_only"], default="all")
    parser.add_argument("--execution-mode", type=str, choices=["eager", "compiled"], default="compiled")
    parser.add_argument("--input-mode", type=str, choices=["repeated", "dataset"], default="repeated")
    parser.add_argument("--dataset-split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--init-from-pytorch-reference", action="store_true")
    parser.add_argument("--pytorch-checkpoint-source", type=str, choices=["base", "sft", "rl"], default=None)
    parser.add_argument("--pytorch-model-tag", type=str, default=None)
    parser.add_argument("--pytorch-step", type=int, default=None)
    add_logging_args(parser)
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
        matrix_optimizer=args.matrix_optimizer,
        muon_ns_steps=args.muon_ns_steps,
        muon_orthogonalize_dtype=args.muon_orthogonalize_dtype,
        muon_block_groups=args.muon_block_groups,
    )
    input_provider = make_input_batch_provider(
        args.input_mode,
        args.device_batch_size,
        args.max_seq_len,
        config.vocab_size,
        bos_token_id,
        dataset_split=args.dataset_split,
    )
    input_metadata = None
    warmup_loss_and_grad = build_loss_and_grad(model)

    warmup_steps_applied = max(args.warmup_steps, 0)
    if args.execution_mode == "compiled" and warmup_steps_applied == 0:
        warmup_steps_applied = 1

    for _ in range(warmup_steps_applied):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        warm_loss, warm_grads = warmup_loss_and_grad(inputs, targets)
        optimizer.update(model, warm_grads)
        eval_training_state(warm_loss, model, optimizer, warm_grads)

    train_step = make_training_step(model, optimizer, execution_mode=args.execution_mode)
    compile_warmup_steps_applied = 0
    if args.execution_mode == "compiled":
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        loss, grads = train_step(inputs, targets)
        eval_training_state(loss, model, optimizer, grads)
        compile_warmup_steps_applied = 1

    mx.reset_peak_memory()
    per_step = []
    for step_index in range(args.steps):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        start = time.perf_counter()
        if args.execution_mode == "compiled":
            loss, grads = train_step(inputs, targets)
            after_backward = start
            after_update = time.perf_counter()
        else:
            loss, grads = train_step(inputs, targets)
            after_backward = time.perf_counter()
            optimizer.update(model, grads)
            after_update = time.perf_counter()
        grad_l2 = tree_l2_norm(grads)
        grad_nonfinite = tree_nonfinite_count(grads)
        param_l2 = tree_l2_norm(model.parameters())
        param_nonfinite = tree_nonfinite_count(model.parameters())
        eval_training_state(loss, model, optimizer, grads)
        after_eval = time.perf_counter()
        elapsed = after_eval - start
        memory = get_memory_stats()
        per_step.append({
            "step": step_index + 1,
            "loss": float(loss.item()),
            "step_time_s": elapsed,
            "tokens_per_s": (args.device_batch_size * args.max_seq_len) / elapsed if elapsed > 0 else 0.0,
            "forward_backward_s": after_update - start if args.execution_mode == "compiled" else after_backward - start,
            "optimizer_update_s": 0.0 if args.execution_mode == "compiled" else after_update - after_backward,
            "eval_s": after_eval - after_update,
            "grad_l2": grad_l2,
            "grad_nonfinite": grad_nonfinite,
            "param_l2": param_l2,
            "param_nonfinite": param_nonfinite,
            "input_batch": batch_metadata,
            "memory": memory,
        })
        if args.progress:
            print(
                f"step {step_index + 1}/{args.steps} loss={per_step[-1]['loss']:.4f} tok/s={per_step[-1]['tokens_per_s']:.1f} grad_l2={per_step[-1]['grad_l2']:.3f} peak_gb={memory['peak_gb']:.2f}",
                flush=True,
            )

    losses = [row["loss"] for row in per_step]
    step_times = [row["step_time_s"] for row in per_step]
    steady_state_step_times = step_times[1:] if len(step_times) > 1 else step_times
    tokens_per_s = [row["tokens_per_s"] for row in per_step]
    forward_backward_times = [row["forward_backward_s"] for row in per_step]
    optimizer_update_times = [row["optimizer_update_s"] for row in per_step]
    eval_times = [row["eval_s"] for row in per_step]
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
            "execution_mode": args.execution_mode,
        },
        "initialization": init_metadata,
        "optimizer": optimizer.metadata(),
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "input_batch": input_metadata,
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
            "warmup_steps_requested": args.warmup_steps,
            "warmup_steps_applied": warmup_steps_applied,
            "compile_warmup_steps_applied": compile_warmup_steps_applied,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "loss_drop_pct": ((losses[0] - losses[-1]) / losses[0]) * 100.0 if losses[0] != 0 else 0.0,
            "mean_step_time_s": statistics.fmean(step_times),
            "mean_forward_backward_s": statistics.fmean(forward_backward_times),
            "mean_optimizer_update_s": statistics.fmean(optimizer_update_times),
            "mean_eval_s": statistics.fmean(eval_times),
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
    log_path = write_summary_log(
        summary,
        log_dir=args.log_dir,
        script_name="mlx_training_check",
        log_prefix=args.log_prefix,
        depth=args.depth,
        input_mode=args.input_mode,
    )
    summary["logging"] = {
        "log_dir": args.log_dir,
        "log_path": log_path,
    }
    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()