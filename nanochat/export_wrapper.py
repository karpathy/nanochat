"""
Export-friendly wrapper for the GPT model.

This module provides a simplified interface for exporting the model to
TorchScript and ONNX formats. It handles:
- Rotary embeddings (embedded in the model)
- Simplified forward pass without Engine complexity
- Optional KV cache for autoregressive generation

Note: Tool use (calculator) and special token handling are not included
in the exported model. These features require Python runtime logic.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ExportableGPT(nn.Module):
    """
    Export-friendly wrapper around the GPT model.
    
    This wrapper provides a simplified forward pass that can be exported
    to TorchScript or ONNX. It includes rotary embeddings and supports
    both single-step and multi-step inference.
    
    Args:
        model: The original GPT model to wrap
        max_seq_len: Maximum sequence length for rotary embeddings
    """
    
    def __init__(self, model, max_seq_len: int = 4096):
        super().__init__()
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        
        # Pre-compute rotary embeddings for the maximum sequence length
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(max_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Pre-compute rotary embeddings."""
        device = self.model.get_device()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            position_offset: Optional position offset for KV cache usage.
                           If provided, should be a scalar tensor indicating
                           the starting position in the sequence.
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        B, T = input_ids.size()
        
        # Determine position offset for rotary embeddings
        if position_offset is None:
            T0 = 0
        else:
            T0 = position_offset.item() if position_offset.dim() == 0 else position_offset[0].item()
        
        # Get rotary embeddings for current sequence
        cos_sin = (
            self.cos[:, T0:T0+T, :, :],
            self.sin[:, T0:T0+T, :, :]
        )
        
        # Forward through the model (without KV cache for simplicity)
        x = self.model.transformer.wte(input_ids)
        x = self._norm(x)
        
        for block in self.model.transformer.h:
            x = x + self._attn_forward(block.attn, self._norm(x), cos_sin)
            x = x + block.mlp(self._norm(x))
        
        x = self._norm(x)
        
        # Compute logits with softcap
        logits = self.model.lm_head(x)
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits
    
    def _norm(self, x):
        """RMS normalization."""
        return torch.nn.functional.rms_norm(x, (x.size(-1),))
    
    def _apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embeddings."""
        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).to(x.dtype)
    
    def _attn_forward(self, attn, x, cos_sin):
        """Simplified attention forward without KV cache."""
        B, T, C = x.size()
        
        # Project to Q, K, V
        q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
        
        # Apply rotary embeddings and normalization
        cos, sin = cos_sin
        q = self._apply_rotary_emb(q, cos, sin)
        k = self._apply_rotary_emb(k, cos, sin)
        q = self._norm(q)
        k = self._norm(k)
        
        # Transpose for attention: (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        enable_gqa = attn.n_head != attn.n_kv_head
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = attn.c_proj(y)
        
        return y


class ExportableGPTWithCache(nn.Module):
    """
    Export-friendly GPT model with explicit KV cache management.
    
    This version maintains KV cache as explicit inputs/outputs, making it
    suitable for stateful inference in C++/ONNX Runtime.
    
    Note: This is more complex and may have limited ONNX support due to
    dynamic shapes. For simplest export, use ExportableGPT without cache.
    """
    
    def __init__(self, model, max_seq_len: int = 4096, max_batch_size: int = 1):
        super().__init__()
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        
        # Pre-compute rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(max_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Pre-compute rotary embeddings."""
        device = self.model.get_device()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def forward(
        self,
        input_ids: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit KV cache.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            cache_k: Key cache (n_layers, batch_size, n_kv_head, max_seq_len, head_dim)
            cache_v: Value cache (n_layers, batch_size, n_kv_head, max_seq_len, head_dim)
            position: Current position in sequence (scalar or batch_size,)
        
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            cache_k: Updated key cache
            cache_v: Updated value cache
        """
        B, T = input_ids.size()
        n_layers = self.config.n_layer
        n_kv_head = self.config.n_kv_head
        head_dim = self.config.n_embd // self.config.n_head
        
        # Initialize cache if not provided
        if cache_k is None:
            cache_k = torch.zeros(
                n_layers, B, n_kv_head, self.max_seq_len, head_dim,
                dtype=torch.bfloat16, device=input_ids.device
            )
        if cache_v is None:
            cache_v = torch.zeros(
                n_layers, B, n_kv_head, self.max_seq_len, head_dim,
                dtype=torch.bfloat16, device=input_ids.device
            )
        if position is None:
            position = torch.tensor(0, dtype=torch.long, device=input_ids.device)
        
        # Get position offset
        T0 = position.item() if position.dim() == 0 else position[0].item()
        
        # Get rotary embeddings
        cos_sin = (
            self.cos[:, T0:T0+T, :, :],
            self.sin[:, T0:T0+T, :, :]
        )
        
        # Forward through transformer
        x = self.model.transformer.wte(input_ids)
        x = self._norm(x)
        
        for layer_idx, block in enumerate(self.model.transformer.h):
            # Attention with cache update
            attn_out, new_k, new_v = self._attn_forward_with_cache(
                block.attn, self._norm(x), cos_sin,
                cache_k[layer_idx], cache_v[layer_idx], T0, T
            )
            x = x + attn_out
            
            # Update cache
            cache_k[layer_idx, :, :, T0:T0+T, :] = new_k
            cache_v[layer_idx, :, :, T0:T0+T, :] = new_v
            
            # MLP
            x = x + block.mlp(self._norm(x))
        
        x = self._norm(x)
        
        # Compute logits
        logits = self.model.lm_head(x)
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits, cache_k, cache_v
    
    def _norm(self, x):
        """RMS normalization."""
        return torch.nn.functional.rms_norm(x, (x.size(-1),))
    
    def _apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embeddings."""
        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).to(x.dtype)
    
    def _attn_forward_with_cache(self, attn, x, cos_sin, cache_k, cache_v, pos, seq_len):
        """Attention forward with cache."""
        B, T, C = x.size()
        
        # Project
        q = attn.c_q(x).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(x).view(B, T, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(x).view(B, T, attn.n_kv_head, attn.head_dim)
        
        # Apply rotary and norm
        cos, sin = cos_sin
        q = self._apply_rotary_emb(q, cos, sin)
        k = self._apply_rotary_emb(k, cos, sin)
        q = self._norm(q)
        k = self._norm(k)
        
        # Transpose
        q = q.transpose(1, 2)
        k_new = k.transpose(1, 2)
        v_new = v.transpose(1, 2)
        
        # Concatenate with cache
        # cache_k/v are (B, H, max_seq_len, D), we need (B, H, pos, D) from cache
        if pos > 0:
            k_cached = cache_k[:, :, :pos, :]
            v_cached = cache_v[:, :, :pos, :]
            k_full = torch.cat([k_cached, k_new], dim=2)
            v_full = torch.cat([v_cached, v_new], dim=2)
        else:
            k_full = k_new
            v_full = v_new
        
        # Attention
        enable_gqa = attn.n_head != attn.n_kv_head
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=True, enable_gqa=enable_gqa
        )
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = attn.c_proj(y)
        
        return y, k_new, v_new
