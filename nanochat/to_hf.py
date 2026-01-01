"""
Convert a nanochat-MoE checkpoint into a HuggingFace-style folder.

Usage (example):
python -m nanochat.to_hf --source base --output hf-export/moe

Notes
- Assumes checkpoints live under ~/.cache/nanochat/<source>_checkpoints/ (same as training scripts).
- The exported model can be loaded with transformers via:
    AutoModelForCausalLM.from_pretrained(<export_dir>, trust_remote_code=True)
- KV cache is not implemented in the HF wrapper; generation works but is not incremental.
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional

import torch
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.generation.utils import GenerationMixin
except ImportError as exc:
    raise SystemExit(
        "transformers is required for HF export. Run `uv sync` (with the hf extra) first."
    ) from exc

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

# standard.py expects `manager` to be importable; alias the package module to satisfy that import.
import nanochat.manager as _manager
sys.modules.setdefault("manager", _manager)
from nanochat.standard import GPT, GPTConfig

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

CHECKPOINT_DIRS = {
    "base": "base_checkpoints",
    "mid": "mid_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}

GPT_CONFIG_FIELDS = {field.name for field in dataclass_fields(GPTConfig)}
HF_CONFIG_FIELDS = {
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "n_exp",
    "top_k",
    "use_aux_loss",
    "use_router_z_loss",
    "use_noisy_top_k",
    "aux_loss_weight",
    "router_z_loss_weight",
    "train_capacity",
    "eval_capacity",
    "min_capacity",
    "stride",
    "use_switch_tfm_init",
    "switch_tfm_init_scale",
    "router_use_full_prec",
}


def load_export_tokenizer(tokenizer_mode: str):
    if tokenizer_mode == "gpt2":
        return RustBPETokenizer.from_pretrained("gpt2")
    if tokenizer_mode == "cache":
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        return RustBPETokenizer.from_directory(tokenizer_dir)
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


def find_largest_model(checkpoint_root: str) -> str:
    model_tags = [name for name in os.listdir(checkpoint_root) if os.path.isdir(os.path.join(checkpoint_root, name))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")
    candidates = []
    for model_tag in model_tags:
        if model_tag.startswith("d") and model_tag[1:].isdigit():
            candidates.append((int(model_tag[1:]), model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_root, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir: str) -> int:
    steps = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("model_") and name.endswith(".pt"):
            stem = name[len("model_"):-3]
            if stem.isdigit():
                steps.append(int(stem))
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)


def normalize_config(cfg_kwargs: dict) -> dict:
    cfg = dict(cfg_kwargs)
    if "sequence_len" in cfg and "block_size" not in cfg:
        cfg["block_size"] = cfg.pop("sequence_len")
    cfg.pop("n_kv_head", None)
    return cfg


def pad_token_embeddings(state_dict: dict, key: str, new_vocab: int):
    if key not in state_dict:
        return
    tensor = state_dict[key]
    if tensor.size(0) >= new_vocab:
        return
    pad_rows = new_vocab - tensor.size(0)
    pad = torch.zeros((pad_rows, tensor.size(1)), device=tensor.device, dtype=tensor.dtype)
    state_dict[key] = torch.cat([tensor, pad], dim=0)


def load_moe_checkpoint(
    source: str,
    model_tag: Optional[str],
    step: Optional[int],
    device: torch.device,
    tokenizer,
):
    base_dir = get_base_dir()
    checkpoint_root = os.path.join(base_dir, CHECKPOINT_DIRS[source])
    if model_tag is None:
        model_tag = find_largest_model(checkpoint_root)
    checkpoint_dir = os.path.join(checkpoint_root, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")

    model_data = torch.load(model_path, map_location=device)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v)
            for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in model_data.items()}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg_kwargs = normalize_config(meta["model_config"])

    tok_vocab = tokenizer.get_vocab_size()
    cfg_vocab = cfg_kwargs.get("vocab_size")
    if cfg_vocab is None:
        cfg_kwargs["vocab_size"] = tok_vocab
    elif tok_vocab > cfg_vocab:
        cfg_kwargs["vocab_size"] = tok_vocab
        pad_token_embeddings(model_data, "transformer.wte.weight", tok_vocab)
        pad_token_embeddings(model_data, "lm_head.weight", tok_vocab)
        meta["model_config"] = cfg_kwargs
    elif tok_vocab < cfg_vocab:
        print(f"[to_hf] tokenizer vocab ({tok_vocab}) < model vocab ({cfg_vocab}); keeping model vocab")

    model_cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in GPT_CONFIG_FIELDS}
    gpt_cfg = GPTConfig(**model_cfg_kwargs)
    model = GPT(gpt_cfg)
    model.load_state_dict(model_data, strict=True)
    model.eval()
    return model, meta, cfg_kwargs


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
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.max_position_embeddings = block_size
        self.n_inner = 4 * n_embd


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
        return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **_: dict,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if labels is None:
            logits, _ = self.model(input_ids)
            loss = None
        else:
            logits, loss = self.model(input_ids, targets=labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}


def write_hf_code(output_dir: str):
    """
    Write the HF-compatible python modules into the export dir.
    These are minimal inference wrappers for trust_remote_code=True.
    """
    config_src = """\
from transformers import PretrainedConfig


class NanoChatMoEHFConfig(PretrainedConfig):
    model_type = "nanochat-moe"

    def __init__(
        self,
        block_size=1024,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,
        n_exp=8,
        top_k=2,
        use_aux_loss=True,
        use_router_z_loss=True,
        use_noisy_top_k=False,
        aux_loss_weight=0.01,
        router_z_loss_weight=0.001,
        train_capacity=1.25,
        eval_capacity=2.0,
        min_capacity=4,
        stride=2,
        use_switch_tfm_init=True,
        switch_tfm_init_scale=1.0,
        router_use_full_prec=True,
        **kwargs,
    ):
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
        # HF aliases
        self.n_positions = block_size
        self.n_ctx = block_size
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.max_position_embeddings = block_size
        self.n_inner = 4 * n_embd
"""

    modeling_src = """\
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_nanochat_moe import NanoChatMoEHFConfig
from .gpt import GPT, GPTConfig


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
        return

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, **_):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        logits, loss = self.model(input_ids, targets=labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}
"""

    tokenizer_src = """\
import os
import pickle
from typing import List

from transformers import PreTrainedTokenizer


class NanoChatTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file: str = "tokenizer.pkl", **kwargs):
        if tokenizer_file is None:
            tokenizer_file = "tokenizer.pkl"
        base_dir = (
            kwargs.pop("pretrained_model_name_or_path", None)
            or kwargs.get("name_or_path", None)
        )
        if base_dir and os.path.isdir(base_dir):
            path = os.path.join(base_dir, tokenizer_file)
        else:
            path = os.path.join(os.path.dirname(__file__), tokenizer_file)

        with open(path, "rb") as f:
            self.enc = pickle.load(f)
        self._vocab_size = int(self.enc.n_vocab)

        specials = set(getattr(self.enc, "special_tokens_set", []))
        if not specials:
            specials = set(getattr(self.enc, "_special_tokens", {}).keys())
        if "<|bos|>" in specials:
            bos = "<|bos|>"
        elif "<|endoftext|>" in specials:
            bos = "<|endoftext|>"
        else:
            raise ValueError(f"Tokenizer specials missing BOS token: {sorted(specials)}")
        eos = "<|assistant_end|>" if "<|assistant_end|>" in specials else bos
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        super().__init__(bos_token=None, eos_token=None, pad_token=None, **kwargs)
        self.added_tokens_encoder.clear()
        self.added_tokens_decoder.clear()
        self._bos_token = bos
        self._eos_token = eos
        self._pad_token = bos
        self._bos_token_id = self.enc._special_tokens[bos]
        self._eos_token_id = self.enc._special_tokens[eos]
        self._pad_token_id = self.enc._special_tokens[bos]

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def get_vocab(self):
        return {str(i): i for i in range(self._vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        ids = self.enc.encode(text, allowed_special=self.enc.special_tokens_set)
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if isinstance(token, str) and token in self.enc._special_tokens:
            return self.enc._special_tokens[token]
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        specials = {v: k for k, v in self.enc._special_tokens.items()}
        if index in specials:
            return specials[index]
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        specials = {v: k for k, v in self.enc._special_tokens.items()}
        parts: List[str] = []
        pending_ids: List[int] = []

        def flush_pending():
            nonlocal pending_ids
            if pending_ids:
                parts.append(self.enc.decode(pending_ids))
                pending_ids.clear()

        for t in tokens:
            if isinstance(t, str) and t in self.enc._special_tokens:
                flush_pending()
                parts.append(t)
            else:
                pending_ids.append(int(t))
        flush_pending()
        return "".join(parts)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))
"""

    gpt_src = """\
import math
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert 1 <= self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None

    def forward(self, x):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        ctx = torch.amp.autocast(device_type=device_type, enabled=False) if self.router_use_full_prec else nullcontext()
        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T
            logits = self.w_g(x)
            if self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
            router_probs = torch.full_like(logits, float("-inf"))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)
            exp_capacity = self.get_capacity(num_tokens)
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)
            exp_mask = exp_mask.permute(1, 0, 2)
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)
            exp_mask *= torch.lt(exp_rank, exp_capacity)
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]
            exp_weights = exp_mask * router_probs
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity)
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool()
            return cb_weight, sec_mask

    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        return int(capacity)


class MLPExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bias = config.bias
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x


class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        self.experts = MLPExperts(config)

    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T
        exp_weight, exp_mask = self.router(x)
        x = x.view(num_tokens, n_embd)
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        exp_out = self.experts(exp_batches)
        exp_weight = exp_weight.view(num_tokens, -1)
        exp_out = exp_out.view(-1, n_embd)
        output = exp_weight.type_as(exp_out) @ exp_out
        return output.view(B, T, n_embd)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MOELayer(config) if use_moe else MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    n_exp: int = 8
    top_k: int = 2
    use_aux_loss: bool = True
    use_router_z_loss: bool = True
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    train_capacity: float = 1.25
    eval_capacity: float = 2.0
    min_capacity: int = 4
    stride: int = 2
    use_switch_tfm_init: bool = True
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        blocks = []
        for i in range(config.n_layer):
            use_moe = (config.n_exp > 1) and (i % config.stride == 0)
            blocks.append(Block(config, use_moe=use_moe))
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(blocks),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
"""

    targets = {
        "configuration_nanochat_moe.py": config_src,
        "modeling_nanochat_moe.py": modeling_src,
        "tokenization_nanochat.py": tokenizer_src,
        "gpt.py": gpt_src,
    }
    for name, content in targets.items():
        with open(Path(output_dir) / name, "w", encoding="utf-8") as f:
            f.write(content)


def export_to_hf(
    source: str,
    output_dir: str,
    model_tag: Optional[str],
    step: Optional[int],
    tokenizer_mode: str,
):
    device = torch.device("cpu")
    tokenizer = load_export_tokenizer(tokenizer_mode)
    model, meta, cfg_kwargs = load_moe_checkpoint(source, model_tag, step, device, tokenizer)
    hf_kwargs = {k: v for k, v in cfg_kwargs.items() if k in HF_CONFIG_FIELDS}
    hf_config = NanoChatMoEHFConfig(**hf_kwargs)
    hf_model = NanoChatMoEHFForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict(), strict=True)

    os.makedirs(output_dir, exist_ok=True)
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_nanochat_moe.NanoChatMoEHFConfig",
        "AutoModelForCausalLM": "modeling_nanochat_moe.NanoChatMoEHFForCausalLM",
        "AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None],
    }
    hf_model.config.tokenizer_class = "NanoChatTokenizer"
    hf_model.config.architectures = ["NanoChatMoEHFForCausalLM"]

    bos_token, eos_token, pad_token, chat_template = resolve_special_tokens(tokenizer)
    hf_model.config.bos_token_id = tokenizer.encode_special(bos_token)
    hf_model.config.eos_token_id = tokenizer.encode_special(eos_token)
    hf_model.config.pad_token_id = tokenizer.encode_special(pad_token)

    if hasattr(hf_model, "generation_config"):
        hf_model.generation_config.top_k = None

    hf_model.save_pretrained(output_dir, safe_serialization=False)
    export_tokenizer_files(output_dir, tokenizer, tokenizer_mode)

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
