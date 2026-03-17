from __future__ import annotations

import json

import mlx.core as mx
import torch

from dev.mlx_checkpoint_translation import apply_pytorch_state_dict_to_mlx_model, build_small_reference_pair


def main() -> None:
    mlx_model, torch_model, token_rows = build_small_reference_pair()
    state_dict = {key: value.detach().cpu() for key, value in torch_model.state_dict().items()}
    apply_pytorch_state_dict_to_mlx_model(mlx_model, state_dict)

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
        "config": {
            "depth": 2,
            "batch_size": len(token_rows),
            "seq_len": len(token_rows[0]),
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