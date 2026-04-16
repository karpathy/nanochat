"""
Inference-only port of nanochat/gpt.py.

Matches the actual nanochat GPT architecture used by d24 SFT checkpoints:
- Smear gate (cheap bigram mixing)
- Backout (mid-layer residual subtraction)
- Per-layer value embeddings (alternating layers, last layer always)
- ve_gate per layer with value embedding
- Sliding-window attention (window_pattern, e.g. "SSSL"), via SDPA
- Rotary embeddings with base=100000, split-halves layout
- Padded vocab (multiple of 64)
- Softcap on logits
- No KV cache (naive autoregressive generate is fine for short responses)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 1536
    window_pattern: str = "SSSL"


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Layers with a value embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        # (B, T, H, D) layout
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = _norm(q), _norm(k)
        q = q * 1.2
        k = k * 1.2

        # SDPA wants (B, H, T, D)
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)

        window = window_size[0]
        if window < 0 or window >= T:
            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True, enable_gqa=enable_gqa)
        else:
            # Sliding window mask (left=window)
            device = q_sdpa.device
            row_idx = torch.arange(T, device=device).unsqueeze(1)
            col_idx = torch.arange(T, device=device).unsqueeze(0)
            mask = (col_idx <= row_idx) & ((row_idx - col_idx) <= window)
            y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(_norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(_norm(x))
        return x


def _compute_window_sizes(config: GPTConfig):
    pattern = config.window_pattern.upper()
    long_window = config.sequence_len
    short_window = -(-long_window // 4 // 128) * 128
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    sizes[-1] = (long_window, 0)
    return sizes


def _precompute_rotary(seq_len, head_dim, base=100000, device="cpu", dtype=torch.float32):
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype)[None, :, None, :]
    return cos, sin


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = _compute_window_sizes(config)

        padded = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(padded, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        )

        # Rotary buffers (registered non-persistent — recomputed in init_rotary)
        self.rotary_seq_len = config.sequence_len * 10
        self.register_buffer("cos", torch.zeros(1), persistent=False)
        self.register_buffer("sin", torch.zeros(1), persistent=False)

    @classmethod
    def from_state_dict(cls, config: GPTConfig, state_dict: dict):
        # Architecture is fixed for this checkpoint family; kept for API compat.
        return cls(config)

    def init_rotary(self, device, dtype):
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = _precompute_rotary(self.rotary_seq_len, head_dim, base=100000, device=device, dtype=dtype)
        self.cos = cos
        self.sin = sin

    # Kept for compatibility with serve.py's existing init_weights() call.
    def init_weights(self):
        pass

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = _norm(x)

        # Smear: bigram mixing (training/prefill path; T >= 1 — guarded for T==1)
        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = _norm(x)

        softcap = 15.0
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits
