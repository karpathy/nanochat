from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn

from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
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


def build_reference_batch(batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None) -> tuple[mx.array, mx.array]:
    seed_ids = [
        bos_token_id if bos_token_id is not None else 1,
        17,
        29,
        113,
        509,
        997,
        4093,
        8191,
    ]
    seed_ids = [token_id % vocab_size for token_id in seed_ids]
    repeated = (seed_ids * ((seq_len // len(seed_ids)) + 1))[:seq_len]
    batch = mx.array([repeated for _ in range(batch_size)], dtype=mx.int32)
    return batch, batch


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
    )
    inputs, targets = build_reference_batch(args.device_batch_size, args.max_seq_len, config.vocab_size, bos_token_id)

    loss_and_grad = nn.value_and_grad(model, lambda batch, labels: model.loss(batch, labels))

    for _ in range(max(args.warmup_steps, 0)):
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), *optimizer.state_trees())

    mx.reset_peak_memory()
    start = time.perf_counter()
    final_loss = None
    for _ in range(args.steps):
        loss, grads = loss_and_grad(inputs, targets)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), *optimizer.state_trees())
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
        },
        "memory": get_memory_stats(),
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "initialization": init_metadata,
        "prototype_limitations": [
            "Uses grouped AdamW instead of reproducing the full MuonAdamW split.",
            "Uses shared-tokenizer-derived synthetic batches instead of the full dataset pipeline.",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()