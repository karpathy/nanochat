from __future__ import annotations

import argparse
import json

import mlx.core as mx
import torch

from dev.mlx_checkpoint_translation import apply_pytorch_state_dict_to_mlx_model
from dev.mlx_gpt_prototype import MLXGPTConfig, MLXGPTPrototype
from nanochat.checkpoint_manager import load_model


def build_token_rows(batch_size: int, seq_len: int, vocab_size: int) -> list[list[int]]:
    rows = []
    for batch_idx in range(batch_size):
        base = 17 * (batch_idx + 1)
        row = [((base + (position * 97)) % (vocab_size - 1)) + 1 for position in range(seq_len)]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch checkpoint logits against translated MLX logits")
    parser.add_argument("--source", type=str, choices=["base", "sft", "rl"], required=True)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    torch_model, _, meta_data = load_model(
        args.source,
        device=torch.device("cpu"),
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    model_config = meta_data["model_config"]
    mlx_config = MLXGPTConfig(**model_config)
    mlx_model = MLXGPTPrototype(mlx_config)

    state_dict = {
        key.removeprefix("_orig_mod."): value.detach().cpu()
        for key, value in torch_model.state_dict().items()
    }
    apply_pytorch_state_dict_to_mlx_model(mlx_model, state_dict)

    seq_len = min(args.seq_len, mlx_config.sequence_len)
    token_rows = build_token_rows(args.batch_size, seq_len, mlx_config.vocab_size)

    torch_inputs = torch.tensor(token_rows, dtype=torch.long)
    torch_targets = torch_inputs.clone()
    mlx_inputs = mx.array(token_rows, dtype=mx.int32)
    mlx_targets = mlx_inputs

    with torch.inference_mode():
        torch_logits = torch_model(torch_inputs)
        torch_loss = torch_model(torch_inputs, torch_targets)

    mlx_logits = mlx_model(mlx_inputs)
    mlx_loss = mlx_model.loss(mlx_inputs, mlx_targets)
    mx.eval(mlx_logits, mlx_loss)

    torch_logits_np = torch_logits.detach().cpu().numpy()
    mlx_logits_np = mx.array(mlx_logits)
    max_abs_diff = float(abs(torch_logits_np - mlx_logits_np).max())
    mean_abs_diff = float(abs(torch_logits_np - mlx_logits_np).mean())

    summary = {
        "checkpoint": {
            "source": args.source,
            "model_tag": args.model_tag,
            "step": args.step,
        },
        "config": {
            "depth": mlx_config.n_layer,
            "batch_size": len(token_rows),
            "seq_len": seq_len,
            "vocab_size": mlx_config.vocab_size,
        },
        "translation": {
            "max_abs_logit_diff": max_abs_diff,
            "mean_abs_logit_diff": mean_abs_diff,
            "torch_loss": float(torch_loss.item()),
            "mlx_loss": float(mlx_loss.item()),
            "loss_abs_diff": abs(float(torch_loss.item()) - float(mlx_loss.item())),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()