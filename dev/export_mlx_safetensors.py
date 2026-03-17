from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from dev.benchmark_mlx_reference import load_tokenizer_metadata
from dev.mlx_checkpoint_translation import initialize_mlx_from_checkpoint_source, initialize_mlx_from_pytorch_reference
from dev.mlx_gpt_prototype import MLXGPTPrototype, build_reference_config


def flatten_tree(tree, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, dict):
        flattened = {}
        for key, value in tree.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_tree(value, next_prefix))
        return flattened
    if isinstance(tree, list):
        flattened = {}
        for index, value in enumerate(tree):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_tree(value, next_prefix))
        return flattened
    if isinstance(tree, tuple):
        flattened = {}
        for index, value in enumerate(tree):
            next_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(flatten_tree(value, next_prefix))
        return flattened
    if prefix == "":
        return {}
    return {prefix: np.asarray(tree)}


def build_init_metadata(model: MLXGPTPrototype, args) -> dict[str, object] | None:
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


def default_export_stem(args) -> str:
    if args.pytorch_checkpoint_source is not None:
        model_tag = args.pytorch_model_tag or "auto"
        step = "latest" if args.pytorch_step is None else f"step{args.pytorch_step}"
        return f"mlx_{args.pytorch_checkpoint_source}_{model_tag}_{step}"
    if args.init_from_pytorch_reference:
        return f"mlx_reference_d{args.depth}"
    return f"mlx_random_d{args.depth}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLX prototype weights and metadata to safetensors for Apple-native runtime work")
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="L")
    parser.add_argument("--init-from-pytorch-reference", action="store_true")
    parser.add_argument("--pytorch-checkpoint-source", type=str, choices=["base", "sft", "rl"], default=None)
    parser.add_argument("--pytorch-model-tag", type=str, default=None)
    parser.add_argument("--pytorch-step", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/mlx_exports")
    parser.add_argument("--output-stem", type=str, default=None)
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
    init_metadata = build_init_metadata(model, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or default_export_stem(args)
    safetensors_path = output_dir / f"{output_stem}.safetensors"
    metadata_path = output_dir / f"{output_stem}.json"

    tensors = flatten_tree(model.parameters())
    safetensors_metadata = {
        "format": "nanochat-mlx-prototype",
        "sequence_len": str(config.sequence_len),
        "vocab_size": str(config.vocab_size),
        "n_layer": str(config.n_layer),
        "n_head": str(config.n_head),
        "n_kv_head": str(config.n_kv_head),
        "n_embd": str(config.n_embd),
        "window_pattern": config.window_pattern,
        "shared_vocab_used": str(shared_tokenizer_used),
        "bos_token_id": "none" if bos_token_id is None else str(bos_token_id),
    }
    save_file(tensors, str(safetensors_path), metadata=safetensors_metadata)

    sidecar = {
        "export": {
            "format": "nanochat-mlx-prototype",
            "tensor_count": len(tensors),
            "safetensors_path": str(safetensors_path),
        },
        "config": asdict(config),
        "tokenizer": {
            "shared_vocab_used": shared_tokenizer_used,
            "bos_token_id": bos_token_id,
        },
        "initialization": init_metadata,
        "tensor_names": sorted(tensors.keys()),
    }
    metadata_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()