"""
Import a Hugging Face model repo into nanochat's native checkpoint format.

This is intended for base-model continuation before multi-stage nanochat runs.

Examples:
python -m scripts.import_hf_checkpoint --repo-id ManmohanSharma/nanochat-d24 --model-tag d24_hf_import
python -m scripts.import_hf_checkpoint --repo-id ManmohanSharma/nanochat-d24 --local-dir /path/to/snapshot
"""

import argparse
import os
from dataclasses import asdict

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanochat.checkpoint_manager import save_checkpoint
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.tools import DEFAULT_TOOL_SCHEMA


def normalize_hf_state_dict_keys(state_dict):
    normalized = {}
    prefixes = ("_orig_mod.", "module.", "model.")
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        normalized[normalized_key] = value
    return normalized


def infer_gpt_config(hf_config):
    kwargs = {
        "sequence_len": getattr(
            hf_config,
            "sequence_len",
            getattr(hf_config, "max_position_embeddings", getattr(hf_config, "n_positions", 2048)),
        ),
        "vocab_size": getattr(hf_config, "vocab_size"),
        "n_layer": getattr(
            hf_config,
            "n_layer",
            getattr(hf_config, "num_hidden_layers", getattr(hf_config, "num_layers")),
        ),
        "n_head": getattr(
            hf_config,
            "n_head",
            getattr(hf_config, "num_attention_heads", None),
        ),
        "n_kv_head": getattr(
            hf_config,
            "n_kv_head",
            getattr(hf_config, "num_key_value_heads", getattr(hf_config, "num_attention_heads", None)),
        ),
        "n_embd": getattr(
            hf_config,
            "n_embd",
            getattr(hf_config, "hidden_size", getattr(hf_config, "d_model", None)),
        ),
        "window_pattern": getattr(hf_config, "window_pattern", "L"),
    }
    missing = [key for key, value in kwargs.items() if value is None]
    if missing:
        raise ValueError(f"Could not infer nanochat GPTConfig fields from HF config: {missing}")
    return GPTConfig(**kwargs)


def verify_tokenizer_compatibility(hf_tokenizer, nanochat_tokenizer):
    hf_vocab = hf_tokenizer.vocab_size
    local_vocab = nanochat_tokenizer.get_vocab_size()
    if hf_vocab != local_vocab:
        raise ValueError(
            f"Tokenizer vocab mismatch: HF repo has vocab_size={hf_vocab}, "
            f"local nanochat tokenizer has vocab_size={local_vocab}"
        )


def load_hf_snapshot(repo_id, revision, token, local_dir):
    if local_dir is not None:
        return local_dir
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        repo_type="model",
    )


def main():
    parser = argparse.ArgumentParser(description="Import HF repo into native nanochat checkpoints")
    parser.add_argument("--repo-id", required=True, help="HF model repo id, e.g. ManmohanSharma/nanochat-d24")
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    parser.add_argument("--local-dir", default=None, help="Use an already-downloaded HF snapshot instead of downloading")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the HF token")
    parser.add_argument("--model-tag", default=None, help="Destination model tag. Defaults to repo name slug")
    parser.add_argument("--step", type=int, default=0, help="Checkpoint step number to write")
    parser.add_argument("--source", choices=["base", "sft", "rl"], default="base", help="Destination checkpoint phase")
    parser.add_argument("--trust-remote-code", type=int, default=1, help="Pass trust_remote_code to Transformers loaders")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    snapshot_path = load_hf_snapshot(args.repo_id, args.revision, token, args.local_dir)
    trust_remote_code = bool(args.trust_remote_code)

    hf_config = AutoConfig.from_pretrained(snapshot_path, token=token, trust_remote_code=trust_remote_code)
    hf_tokenizer = AutoTokenizer.from_pretrained(snapshot_path, token=token, trust_remote_code=trust_remote_code)
    nanochat_tokenizer = get_tokenizer()
    verify_tokenizer_compatibility(hf_tokenizer, nanochat_tokenizer)

    local_config = infer_gpt_config(hf_config)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    hf_state_dict = normalize_hf_state_dict_keys(model.state_dict())

    with torch.device("meta"):
        local_model = GPT(local_config)
    expected_keys = set(local_model.state_dict().keys())
    provided_keys = set(hf_state_dict.keys())
    missing = sorted(expected_keys - provided_keys)
    extra = sorted(provided_keys - expected_keys)
    if missing or extra:
        message = [
            "HF checkpoint keys do not match native nanochat keys after normalization.",
            f"Missing keys: {missing[:12]}",
            f"Extra keys: {extra[:12]}",
        ]
        raise ValueError("\n".join(message))

    model_data = {key: value.detach().cpu() for key, value in hf_state_dict.items()}
    meta_data = {
        "model_config": asdict(local_config),
        "imported_from_hf": True,
        "source_hf_repo": args.repo_id,
        "source_hf_revision": args.revision,
        "tool_schema": DEFAULT_TOOL_SCHEMA,
        "tokenizer_vocab_size": nanochat_tokenizer.get_vocab_size(),
    }

    model_tag = args.model_tag or args.repo_id.split("/")[-1].replace("-", "_")
    base_dir = get_base_dir()
    phase_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[args.source]
    checkpoint_dir = os.path.join(base_dir, phase_dir, model_tag)
    save_checkpoint(checkpoint_dir, args.step, model_data, optimizer_data=None, meta_data=meta_data, rank=0)
    print(f"Imported {args.repo_id} into {checkpoint_dir} at step {args.step}")


if __name__ == "__main__":
    main()
