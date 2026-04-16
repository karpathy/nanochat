"""
Minimal standalone GPT model for Modal inference.
Extracted from nanochat/gpt.py — only the forward-pass code needed for inference.
No training, no DDP, no flash_attention dependency.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 20
    n_head: int = 10
    n_kv_head: int = 10
    n_embd: int = 1280
    window_pattern: str = "L"


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x, offset=0):
        seq_len = x.shape[-2]
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, use_v_emb: bool = False):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.use_v_emb = use_v_emb

        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        if use_v_emb:
            self.v_emb = nn.Parameter(torch.zeros(1, config.n_kv_head, config.sequence_len, self.head_dim))

        self.rotary = RotaryEmbedding(self.head_dim, config.sequence_len)

    def forward(self, x):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary embeddings
        cos, sin = self.rotary(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat k,v if n_kv_head < n_head
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Value embeddings (if enabled)
        if self.use_v_emb:
            v = v + self.v_emb[:, :, :T, :]

        # Scaled dot-product attention (PyTorch native, causal)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, gated: bool = False):
        super().__init__()
        self.gated = gated
        if gated:
            hidden = int(config.n_embd * 8 / 3)
            hidden = ((hidden + 63) // 64) * 64
            self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_fc2 = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            hidden = 4 * config.n_embd
            self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        if self.gated:
            a = self.c_fc(x)
            b = self.c_fc2(x)
            return self.c_proj(F.relu(a).pow(2) * b)
        else:
            return self.c_proj(F.relu(self.c_fc(x)).pow(2))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, gated_mlp: bool = False, use_v_emb: bool = False):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, use_v_emb=use_v_emb)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config, gated=gated_mlp)
        self.layer_idx = layer_idx

    def forward(self, x, resid_lambda=1.0, x0_lambda=0.0, x0=None):
        h = x * resid_lambda + self.attn(self.ln_1(x))
        if x0 is not None and x0_lambda != 0.0:
            h = h + x0_lambda * x0
        h2 = h * resid_lambda + self.mlp(self.ln_2(h))
        if x0 is not None and x0_lambda != 0.0:
            h2 = h2 + x0_lambda * x0
        return h2


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, gated_mlp: bool = False, use_v_emb: bool = False):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            norm_emb=RMSNorm(config.n_embd),
            h=nn.ModuleList([Block(config, i, gated_mlp=gated_mlp, use_v_emb=use_v_emb) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Residual lambdas (per-layer scaling)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

    @classmethod
    def from_state_dict(cls, config: GPTConfig, state_dict: dict):
        """Auto-detect architecture features from checkpoint keys."""
        gated = any("c_fc2" in k for k in state_dict)
        v_emb = any("v_emb" in k for k in state_dict)
        model = cls(config, gated_mlp=gated, use_v_emb=v_emb)
        return model

    def init_weights(self):
        """Initialize rotary embeddings and value embeddings."""
        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                inv_freq = 1.0 / (10000 ** (torch.arange(0, module.inv_freq.shape[0] * 2, 2).float() / (module.inv_freq.shape[0] * 2)))
                module.inv_freq.copy_(inv_freq)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.sequence_len, f"Input length {T} exceeds max {self.config.sequence_len}"

        x = self.transformer.wte(idx)
        x = self.transformer.norm_emb(x)
        x0 = x  # save for residual connections

        for i, block in enumerate(self.transformer.h):
            rl = self.resid_lambdas[i].item()
            xl = self.x0_lambdas[i].item()
            x = block(x, resid_lambda=rl, x0_lambda=xl, x0=x0)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
