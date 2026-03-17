from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn

from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
from dev.mlx_input_batches import make_input_batch_provider
from dev.mlx_logging import add_logging_args, write_summary_log
from dev.mlx_gpt_prototype import MLXGPTPrototype, build_reference_config
from nanochat.tokenizer import get_tokenizer


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def load_tokenizer_metadata(default_vocab_size: int) -> tuple[int, int | None, bool]:
    try:
        tokenizer = get_tokenizer()
        return tokenizer.get_vocab_size(), tokenizer.get_bos_token_id(), True
    except Exception:
        return default_vocab_size, None, False

def get_memory_stats() -> dict[str, float]:
    return {
        "active_gb": bytes_to_gb(mx.get_active_memory()),
        "peak_gb": bytes_to_gb(mx.get_peak_memory()),
        "cache_gb": bytes_to_gb(mx.get_cache_memory()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the MLX GPT reference prototype")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--matrix-optimizer", type=str, choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-orthogonalize-dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--muon-block-groups", type=str, choices=["all", "mlp_only", "attn_only"], default="all")
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
    init_metadata = None
    if args.init_from_pytorch_reference:
        init_metadata = initialize_mlx_from_pytorch_reference(model)
    elif args.pytorch_checkpoint_source is not None:
        init_metadata = initialize_mlx_from_checkpoint_source(
            model,
            source=args.pytorch_checkpoint_source,
            model_tag=args.pytorch_model_tag,
            step=args.pytorch_step,
        )

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

    loss_and_grad = nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))

    for _ in range(max(args.warmup_steps, 0)):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), *optimizer.state_trees())

    mx.reset_peak_memory()
    start = time.perf_counter()
    forward_backward_elapsed = 0.0
    optimizer_update_elapsed = 0.0
    eval_elapsed = 0.0
    final_loss = None
    for _ in range(args.steps):
        inputs, targets, batch_metadata = input_provider.next_batch()
        if input_metadata is None:
            input_metadata = batch_metadata
        step_start = time.perf_counter()
        loss, grads = loss_and_grad(inputs, targets)
        after_backward = time.perf_counter()
        optimizer.update(model, grads)
        after_update = time.perf_counter()
        mx.eval(loss, model.parameters(), *optimizer.state_trees())
        after_eval = time.perf_counter()
        forward_backward_elapsed += after_backward - step_start
        optimizer_update_elapsed += after_update - after_backward
        eval_elapsed += after_eval - after_update
        final_loss = loss
    elapsed = time.perf_counter() - start

    tokens_processed = args.device_batch_size * args.max_seq_len * args.steps
    summary = {
        "config": {
            "depth": config.n_layer,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": config.sequence_len,
            "model_dim": config.n_embd,
            "heads": config.n_head,
            "vocab_size": config.vocab_size,
            "window_pattern": config.window_pattern,
            "params_total": model.num_params(),
        },
        "benchmark": {
            "steps": args.steps,
            "elapsed_s": elapsed,
            "tokens_per_s": tokens_processed / elapsed if elapsed > 0 else 0.0,
            "loss": float(final_loss.item()) if final_loss is not None else None,
            "mean_forward_backward_s": forward_backward_elapsed / args.steps if args.steps > 0 else 0.0,
            "mean_optimizer_update_s": optimizer_update_elapsed / args.steps if args.steps > 0 else 0.0,
            "mean_eval_s": eval_elapsed / args.steps if args.steps > 0 else 0.0,
        },
        "optimizer": optimizer.metadata(),
        "memory": get_memory_stats(),
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "input_batch": input_metadata,
        "initialization": init_metadata,
        "prototype_limitations": [
            "Matrix optimizer parity is still experimental when using the MLX Muon option.",
            "Dataset-backed input mode is still a single-process sequential packer rather than the full distributed best-fit loader.",
        ],
    }
    log_path = write_summary_log(
        summary,
        log_dir=args.log_dir,
        script_name="benchmark_mlx_reference",
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