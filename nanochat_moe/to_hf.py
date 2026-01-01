"""
Convert a nanochat-MoE checkpoint into a HuggingFace-style folder.

Usage (example):
python -m nanochat_moe.to_hf --source base --output hf-export/moe

Notes
- Assumes checkpoints live under ~/.cache/nanochat/<source>_checkpoints/ (same as training scripts).
- The exported model can be loaded with transformers via:
    AutoModelForCausalLM.from_pretrained(<export_dir>, trust_remote_code=True)
- KV cache is not implemented in the HF wrapper; generation works but is not incremental.
"""
import argparse
import os
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import sys
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.generation.utils import GenerationMixin
except ImportError as exc:
    raise SystemExit(
        "transformers is required for HF export. Run `uv sync` (with the hf extra) first."
    ) from exc
    

from nanochat_moe.checkpoint_manager import find_last_step, find_largest_model
# standard.py expects `manager` to be importable; alias the package module to satisfy that import.
import nanochat_moe.manager as _moe_manager
sys.modules.setdefault("manager", _moe_manager)
from nanochat_moe.standard import GPT, GPTConfig
from nanochat_moe.common import get_base_dir
from nanochat_moe.tokenizer import get_tokenizer, RustBPETokenizer

CHAT_TEMPLATE = (
    "<|bos|>{% for message in messages %}{% if message['role'] == 'user' %}"
    "<|user_start|>{{ message['content'] }}<|user_end|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
    "{% endif %}{% endfor %}<|assistant_start|>"
)

CHAT_SPECIAL_TOKENS = {
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
}


def load_export_tokenizer(tokenizer_mode: str):
    if tokenizer_mode == "gpt2":
        return RustBPETokenizer.from_pretrained("gpt2")
    if tokenizer_mode == "cache":
        return get_tokenizer()
    raise ValueError(f"Unknown tokenizer mode: {tokenizer_mode}")


def resolve_special_tokens(tokenizer):
    specials = set(tokenizer.get_special_tokens())
    if "<|bos|>" in specials:
        bos = "<|bos|>"
    elif "<|endoftext|>" in specials:
        bos = "<|endoftext|>"
    else:
        raise ValueError(f"Tokenizer specials missing BOS token: {sorted(specials)}")

    if "<|assistant_end|>" in specials:
        eos = "<|assistant_end|>"
    else:
        eos = bos

    pad = bos
    chat_template = (
        CHAT_TEMPLATE
        if "<|bos|>" in specials and CHAT_SPECIAL_TOKENS.issubset(specials)
        else None
    )
    return bos, eos, pad, chat_template


class NanoChatMoEHFConfig(PretrainedConfig):
    model_type = "nanochat-moe"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.0,
        bias: bool = False,
        n_exp: int = 8,
        top_k: int = 2,
        use_aux_loss: bool = True,
        use_router_z_loss: bool = True,
        use_noisy_top_k: bool = False,
        aux_loss_weight: float = 0.01,
        router_z_loss_weight: float = 0.001,
        train_capacity: float = 1.25,
        eval_capacity: float = 2.0,
        min_capacity: int = 4,
        stride: int = 2,
        use_switch_tfm_init: bool = True,
        switch_tfm_init_scale: float = 1.0,
        router_use_full_prec: bool = True,
        **kwargs,
    ):
        # Don't tie embeddings; nanochat uses untied wte/lm_head
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.n_exp = n_exp
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.use_router_z_loss = use_router_z_loss
        self.use_noisy_top_k = use_noisy_top_k
        self.aux_loss_weight = aux_loss_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.train_capacity = train_capacity
        self.eval_capacity = eval_capacity
        self.min_capacity = min_capacity
        self.stride = stride
        self.use_switch_tfm_init = use_switch_tfm_init
        self.switch_tfm_init_scale = switch_tfm_init_scale
        self.router_use_full_prec = router_use_full_prec
        # HF compatibility aliases
        self.n_positions = block_size
        self.n_ctx = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_inner = 4 * n_embd
        self.max_position_embeddings = block_size


class NanoChatMoEHFForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatMoEHFConfig

    def __init__(self, config: NanoChatMoEHFConfig):
        super().__init__(config)
        gpt_cfg = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            n_exp=config.n_exp,
            top_k=config.top_k,
            use_aux_loss=config.use_aux_loss,
            use_router_z_loss=config.use_router_z_loss,
            use_noisy_top_k=config.use_noisy_top_k,
            aux_loss_weight=config.aux_loss_weight,
            router_z_loss_weight=config.router_z_loss_weight,
            train_capacity=config.train_capacity,
            eval_capacity=config.eval_capacity,
            min_capacity=config.min_capacity,
            stride=config.stride,
            use_switch_tfm_init=config.use_switch_tfm_init,
            switch_tfm_init_scale=config.switch_tfm_init_scale,
            router_use_full_prec=config.router_use_full_prec,
        )
        self.model = GPT(gpt_cfg)

    def get_input_embeddings(self):
        return self.model.transformer.wte

    def set_input_embeddings(self, value):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def tie_weights(self):
        # nanochat uses untied embeddings; override to no-op
        return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # unused
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,  # not implemented
        **_: dict,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            # Align shapes for CE: shift labels to match logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}


def export_tokenizer_files(output_dir: str, tokenizer, tokenizer_mode: str):
    if tokenizer_mode == "cache":
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        if os.path.isdir(tokenizer_dir):
            for name in os.listdir(tokenizer_dir):
                src = os.path.join(tokenizer_dir, name)
                dst = os.path.join(output_dir, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            print(f"[to_hf] Copied tokenizer files from {tokenizer_dir} to {output_dir}")
            return
        print(f"[to_hf] tokenizer directory not found at {tokenizer_dir}, exporting tokenizer.pkl only")
    tokenizer.save(output_dir)


def load_moe_checkpoint(
    source: str,
    model_tag: Optional[str],
    step: Optional[int],
    device: torch.device,
    tokenizer,
):
    """
    Load a nanochat-MoE checkpoint (model_<step>.pt + meta_<step>.json) using the standard GPT architecture.
    """
    base_dir = get_base_dir()
    checkpoint_root = os.path.join(base_dir, f"{source}_checkpoints")
    if model_tag is None:
        model_tag = find_largest_model(checkpoint_root)
    checkpoint_dir = os.path.join(checkpoint_root, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

    model_data = torch.load(model_path, map_location=device)
    if device.type in {"cpu", "mps"}:
        model_data = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v) for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg_kwargs = meta["model_config"]
    # If tokenizer size differs, pad embeddings and update config
    tok_vocab = tokenizer.get_vocab_size()
    cfg_vocab = cfg_kwargs.get("vocab_size")
    if tok_vocab > cfg_vocab:
        cfg_kwargs["vocab_size"] = tok_vocab
        # pad wte and lm_head to match tokenizer size
        for key in ["transformer.wte.weight", "lm_head.weight"]:
            if key in model_data:
                tensor = model_data[key]
                if tensor.size(0) < tok_vocab:
                    pad_rows = tok_vocab - tensor.size(0)
                    pad = torch.zeros((pad_rows, tensor.size(1)), device=tensor.device, dtype=tensor.dtype)
                    model_data[key] = torch.cat([tensor, pad], dim=0)
        meta["model_config"] = cfg_kwargs
    elif tok_vocab < cfg_vocab:
        print(f"[to_hf] tokenizer vocab ({tok_vocab}) < model vocab ({cfg_vocab}); keeping model vocab")
    gpt_cfg = GPTConfig(**cfg_kwargs)
    model = GPT(gpt_cfg)
    model.load_state_dict(model_data, strict=True)
    model.eval()
    return model, meta

def write_hf_code(output_dir: str):
    """
    Copy the HF-compatible python modules into the export dir.
    Prefer the checked-in template under hf-export/std; fallback to writing nothing
    if not present (export will still work if trust_remote_code imports from local FS).
    """
    template_dir = Path(__file__).resolve().parent.parent / "hf-export" / "moe"
    targets = [
        "configuration_nanochat_moe.py",
        "modeling_nanochat_moe.py",
        "tokenization_nanochat.py",
        "gpt.py",
    ]
    if template_dir.is_dir():
        for name in targets:
            src = template_dir / name
            if src.exists():
                shutil.copy2(src, Path(output_dir) / name)
    else:
        print(f"[to_hf] Template dir not found at {template_dir}, skipping python module copy")


def export_to_hf(
    source: str,
    output_dir: str,
    model_tag: Optional[str],
    step: Optional[int],
    tokenizer_mode: str,
):
    device = torch.device("cpu")
    tokenizer = load_export_tokenizer(tokenizer_mode)
    model, meta = load_moe_checkpoint(source, model_tag, step, device, tokenizer)
    cfg_kwargs = meta["model_config"]
    hf_config = NanoChatMoEHFConfig(**cfg_kwargs)
    hf_model = NanoChatMoEHFForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict(), strict=True)

    os.makedirs(output_dir, exist_ok=True)
    # Tell transformers how to load custom code when trust_remote_code=True
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_nanochat_moe.NanoChatMoEHFConfig",
        "AutoModelForCausalLM": "modeling_nanochat_moe.NanoChatMoEHFForCausalLM",
        "AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None],
    }
    hf_model.config.tokenizer_class = "NanoChatTokenizer"
    hf_model.config.architectures = ["NanoChatMoEHFForCausalLM"]
    # Bind special token ids/strings from tokenizer
    bos_token, eos_token, pad_token, chat_template = resolve_special_tokens(tokenizer)
    bos_id = tokenizer.encode_special(bos_token)
    eos_id = tokenizer.encode_special(eos_token)
    pad_id = tokenizer.encode_special(pad_token)
    hf_model.config.bos_token_id = bos_id
    hf_model.config.eos_token_id = eos_id
    hf_model.config.pad_token_id = pad_id
    # generation config safety: clear sampling-only fields when do_sample is False
    if hasattr(hf_model, "generation_config"):
        hf_model.generation_config.top_k = None

    hf_model.save_pretrained(output_dir, safe_serialization=False)
    # Best effort: drop tokenizer files alongside weights
    export_tokenizer_files(output_dir, tokenizer, tokenizer_mode)
    # Write tokenizer_config with chat template and special tokens
    tok_cfg = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
    }
    if chat_template is not None:
        tok_cfg["chat_template"] = chat_template
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_cfg, f, indent=2)
    print(f"[to_hf] Exported {source} checkpoint to {output_dir}")
    write_hf_code(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Export nanochat-MoE checkpoint to HuggingFace format")
    parser.add_argument("--source", choices=["base", "mid", "sft", "rl"], default="base", help="Which checkpoint family to export")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d20). Defaults to largest available.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--output", type=str, default="hf-export/moe", help="Output directory for HF files")
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "cache"],
        default="gpt2",
        help="Tokenizer source for export: gpt2 uses tiktoken; cache uses ~/.cache/nanochat/tokenizer",
    )
    args = parser.parse_args()

    export_to_hf(args.source, args.output, args.model_tag, args.step, args.tokenizer)


if __name__ == "__main__":
    main()
