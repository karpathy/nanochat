from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from dev.benchmark_mlx_reference import get_memory_stats, load_tokenizer_metadata
from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
from dev.mlx_gpt_prototype import MLXGPTPrototype, build_reference_config
from dev.mlx_input_batches import make_input_batch_provider
from dev.mlx_logging import add_logging_args, write_summary_log


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


def build_eager_train_step(model: MLXGPTPrototype, optimizer):
    loss_and_grad = nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))

    def step(batch, labels):
        loss, grads = loss_and_grad(batch, labels)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), *optimizer.state_trees())
        return loss

    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a longer MLX training session on the Apple-native prototype")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=4)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--matrix-optimizer", type=str, choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--input-mode", type=str, choices=["repeated", "dataset"], default="repeated")
    parser.add_argument("--dataset-split", type=str, choices=["train", "val"], default="train")
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

    train_step = build_eager_train_step(model, optimizer)

    bootstrap_loss = None
    for warmup_idx in range(max(args.warmup_steps, 0)):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        warm_loss = train_step(inputs, targets)
        if warmup_idx == 0:
            bootstrap_loss = warm_loss

    mx.reset_peak_memory()
    per_step = []
    wall_start = time.perf_counter()
    for step_idx in range(args.steps):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        start = time.perf_counter()
        loss = train_step(inputs, targets)
        mx.eval(loss)
        elapsed = time.perf_counter() - start
        memory = get_memory_stats()
        row = {
            "step": step_idx + 1,
            "loss": float(loss.item()),
            "step_time_s": elapsed,
            "tokens_per_s": (args.device_batch_size * args.max_seq_len) / elapsed if elapsed > 0 else 0.0,
            "input_batch": batch_metadata,
            "memory": memory,
        }
        per_step.append(row)
        if args.progress_interval > 0 and ((step_idx + 1) % args.progress_interval == 0 or step_idx == 0 or step_idx + 1 == args.steps):
            print(
                f"step {step_idx + 1}/{args.steps} loss={row['loss']:.4f} tok/s={row['tokens_per_s']:.1f} active_gb={memory['active_gb']:.2f} peak_gb={memory['peak_gb']:.2f}",
                flush=True,
            )
    wall_elapsed = time.perf_counter() - wall_start

    losses = [row["loss"] for row in per_step]
    throughputs = [row["tokens_per_s"] for row in per_step]
    step_times = [row["step_time_s"] for row in per_step]

    summary = {
        "config": {
            "depth": config.n_layer,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": config.sequence_len,
            "model_dim": config.n_embd,
            "heads": config.n_head,
            "vocab_size": config.vocab_size,
            "params_total": model.num_params(),
            "optimizer": "grouped_optimizer",
            "matrix_optimizer": args.matrix_optimizer,
            "weight_decay": args.weight_decay,
            "execution_mode": "eager_mlx",
        },
        "initialization": init_metadata,
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "input_batch": input_metadata,
        "session": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "bootstrap_loss": float(bootstrap_loss.item()) if bootstrap_loss is not None else None,
            "wall_elapsed_s": wall_elapsed,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "loss_drop_pct": ((losses[0] - losses[-1]) / losses[0]) * 100.0 if losses[0] != 0 else 0.0,
            "mean_step_time_s": statistics.fmean(step_times),
            "mean_tokens_per_s": statistics.fmean(throughputs),
            "max_peak_memory_gb": max(row["memory"]["peak_gb"] for row in per_step),
            "final_active_memory_gb": per_step[-1]["memory"]["active_gb"],
        },
        "per_step": per_step,
    }
    log_path = write_summary_log(
        summary,
        log_dir=args.log_dir,
        script_name="mlx_training_session",
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