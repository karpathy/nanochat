"""
Attention mechanisms for GPT model.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMS normalization."""
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.

    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to queries or keys."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with rotary embeddings, QK norm, and optional value embeddings."""

    def __init__(self, config: object, layer_idx: int):
        super().__init__()
        self.layer_idx: int = layer_idx
        self.n_head: int = config.n_head
        self.n_kv_head: int = config.n_kv_head
        self.n_embd: int = config.n_embd
        self.head_dim: int = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels: int = 12
        self.ve_gate: Optional[Linear] = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, C)
        ve: Optional[torch.Tensor],  # (B, T, n_kv_head * head_dim) or None
        cos_sin: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) rotary embeddings
        window_size: Tuple[int, int],  # (left, right) for sliding window
        kv_cache: Optional[Any] = None,  # KVCache or None
    ) -> torch.Tensor:  # (B, T, C)
        """Forward pass with optional KV caching for inference."""
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q = q * 1.15  # sharper attention (split scale between Q and K)
        k = k * 1.15

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
