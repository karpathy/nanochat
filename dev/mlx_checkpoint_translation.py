from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx
import torch

from dev.mlx_gpt_prototype import MLXGPTPrototype, MLXGPTConfig, build_reference_config, has_ve
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig


def mlx_config_to_pytorch_config(config: MLXGPTConfig) -> GPTConfig:
    return GPTConfig(
        sequence_len=config.sequence_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_embd=config.n_embd,
        window_pattern=config.window_pattern,
    )


def torch_tensor_to_mx(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.detach().cpu().float().numpy())


def translate_state_dict_to_mlx_tree(state_dict: dict[str, torch.Tensor], config: MLXGPTConfig) -> dict:
    translated = {
        "wte": {"weight": torch_tensor_to_mx(state_dict["transformer.wte.weight"] )},
        "blocks": [],
        "lm_head": {"weight": torch_tensor_to_mx(state_dict["lm_head.weight"])},
        "resid_lambdas": torch_tensor_to_mx(state_dict["resid_lambdas"]),
        "x0_lambdas": torch_tensor_to_mx(state_dict["x0_lambdas"]),
        "value_embeds": [],
    }

    for layer_idx in range(config.n_layer):
        block_prefix = f"transformer.h.{layer_idx}"
        block_tree = {
            "attn": {
                "c_q": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.attn.c_q.weight"])},
                "c_k": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.attn.c_k.weight"])},
                "c_v": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.attn.c_v.weight"])},
                "c_proj": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.attn.c_proj.weight"])},
            },
            "mlp": {
                "c_fc": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.mlp.c_fc.weight"])},
                "c_proj": {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.mlp.c_proj.weight"])},
            },
        }
        if has_ve(layer_idx, config.n_layer):
            block_tree["attn"]["ve_gate"] = {"weight": torch_tensor_to_mx(state_dict[f"{block_prefix}.attn.ve_gate.weight"])}
            translated["value_embeds"].append({"weight": torch_tensor_to_mx(state_dict[f"value_embeds.{layer_idx}.weight"])})
        else:
            translated["value_embeds"].append({})
        translated["blocks"].append(block_tree)

    return translated


def apply_pytorch_state_dict_to_mlx_model(model: MLXGPTPrototype, state_dict: dict[str, torch.Tensor]) -> None:
    translated = translate_state_dict_to_mlx_tree(state_dict, model.config)
    model.update(translated)


def build_matching_pytorch_model(config: MLXGPTConfig) -> GPT:
    torch_config = mlx_config_to_pytorch_config(config)
    model = GPT(torch_config)
    model.init_weights()
    model.eval()
    return model


def initialize_mlx_from_pytorch_reference(model: MLXGPTPrototype) -> dict[str, object]:
    torch_model = build_matching_pytorch_model(model.config)
    state_dict = {key: value.detach().cpu() for key, value in torch_model.state_dict().items()}
    apply_pytorch_state_dict_to_mlx_model(model, state_dict)
    return {
        "source": "pytorch-random-reference",
        "model_config": asdict(mlx_config_to_pytorch_config(model.config)),
    }


def initialize_mlx_from_checkpoint_source(model: MLXGPTPrototype, source: str, model_tag: str | None = None, step: int | None = None) -> dict[str, object]:
    torch_model, _, meta_data = load_model(source, device=torch.device("cpu"), phase="eval", model_tag=model_tag, step=step)
    state_dict = {key.removeprefix("_orig_mod."): value.detach().cpu() for key, value in torch_model.state_dict().items()}
    apply_pytorch_state_dict_to_mlx_model(model, state_dict)
    return {
        "source": source,
        "model_tag": model_tag,
        "step": step,
        "model_config": meta_data["model_config"],
    }


def build_small_reference_pair() -> tuple[MLXGPTPrototype, GPT, list[list[int]]]:
    config = build_reference_config(depth=2, sequence_len=32, aspect_ratio=64, head_dim=128)
    mlx_model = MLXGPTPrototype(config)
    torch_model = build_matching_pytorch_model(config)
    token_rows = [[1, 17, 29, 113, 509, 997, 4093, 8191] * 4]
    return mlx_model, torch_model, token_rows