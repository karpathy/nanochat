"""
Convert a nanochat checkpoint into a HuggingFace-style folder.

Usage (example):
python -m nanochat.to_hf --source base --output hf-export/base

Notes
- Assumes checkpoints live under ~/.cache/nanochat/<source>_checkpoints/ (same as training scripts).
- The exported model can be loaded with transformers via:
    AutoModelForCausalLM.from_pretrained(<export_dir>, trust_remote_code=True)
- KV cache is not implemented in the HF wrapper; generation works but is not incremental.
"""
import argparse
import os
import shutil
from typing import Optional

import torch
import torch.nn.functional as F
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.generation.utils import GenerationMixin
except ImportError as exc:
    raise SystemExit(
        "transformers is required for HF export. Run `uv sync` (with the hf extra) first."
    ) from exc
    

from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


class NanoChatHFConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 6,
        n_kv_head: int = 6,
        n_embd: int = 768,
        **kwargs,
    ):
        # Don't tie embeddings; nanochat uses untied wte/lm_head
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        # HF compatibility aliases
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = sequence_len


class NanoChatHFForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatHFConfig

    def __init__(self, config: NanoChatHFConfig):
        super().__init__(config)
        gpt_cfg = GPTConfig(
            sequence_len=config.sequence_len,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd,
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


def copy_tokenizer_files(output_dir: str):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        print(f"[to_hf] tokenizer directory not found at {tokenizer_dir}, skipping tokenizer export")
        return
    for name in os.listdir(tokenizer_dir):
        src = os.path.join(tokenizer_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[to_hf] Copied tokenizer files from {tokenizer_dir} to {output_dir}")

def write_hf_code(output_dir: str):
    cfg_py = r'''
from transformers import PretrainedConfig

class NanoChatHFConfig(PretrainedConfig):
    model_type = "nanochat"
    def __init__(self, sequence_len=1024, vocab_size=50304, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        # HF compatibility aliases
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = sequence_len
'''
    mdl_py = r'''
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_nanochat import NanoChatHFConfig
from .gpt import GPT, GPTConfig

class NanoChatHFForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatHFConfig

    def __init__(self, config: NanoChatHFConfig):
        super().__init__(config)
        gpt_cfg = GPTConfig(
            sequence_len=config.sequence_len,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd,
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
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}
    '''
    gpt_py = r'''
"""
Minimal GPT implementation for HF export (inference-only utilities).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Precompute rotary embeddings (small overhead, avoids realloc each forward)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get_device(self):
        return self.transformer.wte.weight.device

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        # Avoid meta buffers when HF initializes under init_empty_weights; fall back to CPU.
        if getattr(device, "type", None) == "meta":
            device = torch.device("cpu")
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        softcap = 15
        if targets is not None:
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits
'''
    tok_py = r'''
import os
import pickle
from typing import List, Optional

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

        kwargs.setdefault("bos_token", "<|bos|>")
        # Fallback: reuse bos as eos/pad to satisfy HF APIs; underlying vocab is index-only.
        kwargs.setdefault("eos_token", kwargs["bos_token"])
        kwargs.setdefault("pad_token", kwargs["bos_token"])
        super().__init__(**kwargs)

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return {str(i): i for i in range(self._vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        ids = self.enc.encode_ordinary(text)
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if isinstance(token, str) and token.startswith("<|") and token.endswith("|>"):
            return 0
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ids = []
        for t in tokens:
            try:
                ids.append(int(t))
            except (ValueError, TypeError):
                # Skip special tokens like <|pad|> that are not part of the numeric vocab.
                continue
        return self.enc.decode(ids)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()
'''
    chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|user_start|>{{ message['content'] }}<|user_end|>{% elif message['role'] == 'assistant' %}<|assistant_start|>{{ message['content'] }}<|assistant_end|>{% endif %}{% endfor %}<|assistant_start|>"
    with open(os.path.join(output_dir, "configuration_nanochat.py"), "w") as f:
        f.write(cfg_py)
    with open(os.path.join(output_dir, "modeling_nanochat.py"), "w") as f:
        f.write(mdl_py)
    with open(os.path.join(output_dir, "gpt.py"), "w") as f:
        f.write(gpt_py)
    with open(os.path.join(output_dir, "tokenization_nanochat.py"), "w") as f:
        f.write(tok_py)
    # Minimal tokenizer_config for transformers (adds chat template and tokens).
    import json
    tok_cfg = {
        "chat_template": chat_template
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_cfg, f, indent=2)


def export_to_hf(source: str, output_dir: str, model_tag: Optional[str], step: Optional[int]):
    device = torch.device("cpu")
    model, tokenizer, meta = load_model(source, device=device, phase="eval", model_tag=model_tag, step=step)
    cfg_kwargs = meta["model_config"]
    hf_config = NanoChatHFConfig(**cfg_kwargs)
    hf_model = NanoChatHFForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict(), strict=True)

    os.makedirs(output_dir, exist_ok=True)
        # Tell transformers how to load custom code when trust_remote_code=True
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_nanochat.NanoChatHFConfig",
        "AutoModelForCausalLM": "modeling_nanochat.NanoChatHFForCausalLM",
        "AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None],
    }
    hf_model.config.tokenizer_class = "NanoChatTokenizer"
    hf_model.config.architectures = ["NanoChatHFForCausalLM"]

    hf_model.save_pretrained(output_dir, safe_serialization=False)
    # Best effort: drop tokenizer files alongside weights
    copy_tokenizer_files(output_dir)
    print(f"[to_hf] Exported {source} checkpoint to {output_dir}")
    write_hf_code(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Export nanochat checkpoint to HuggingFace format")
    parser.add_argument("--source", choices=["base", "mid", "sft", "rl"], default="base", help="Which checkpoint family to export")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d20). Defaults to largest available.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--output", type=str, default="hf-export", help="Output directory for HF files")
    args = parser.parse_args()

    export_to_hf(args.source, args.output, args.model_tag, args.step)


if __name__ == "__main__":
    main()
