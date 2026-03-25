"""
Unified Flash Attention interface with three-tier automatic backend selection:

    FA3 (Hopper sm90)  ->  FA2 (Ampere sm80 / Ada sm89)  ->  PyTorch SDPA fallback

Exports `flash_attn` module with two functions:

Usage:
    from nanochat.flash_attention import flash_attn

    # Training with packed variable-length sequences: q, k, v are (total_tokens, H, D)
    y = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)

All non-cached forward passes go through varlen. (B, T) callers that don't provide
cu_seqlens get it auto-constructed in GPT.forward.

FA3 and FA2 both support flash_attn_varlen_func with per-document attention isolation.
The SDPA fallback reshapes to (B, T_seq) and uses is_causal=True -- no doc isolation,
but efficient kernels on all hardware (Mac, CPU, Blackwell, older GPUs).
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try FA3 (Hopper), then FA2 (Ampere/Ada), then SDPA fallback
# =============================================================================
def _load_flash_attention():
    """Try to load Flash Attention kernels. Returns (module, version_string) or (None, None)."""
    if not torch.cuda.is_available():
        return None, None
    try:
        major, _ = torch.cuda.get_device_capability()
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        # FA3: Hopper (sm90) only
        if major == 9:
            try:
                return get_kernel('varunneal/flash-attention-3').flash_attn_interface, 'fa3'
            except Exception:
                pass

        # FA2: Ampere (sm80), Ada (sm89), and Hopper fallback
        if major >= 8:
            try:
                return get_kernel('kernels-community/flash-attn2').flash_attn_interface, 'fa2'
            except Exception:
                pass
    except Exception:
        pass
    return None, None


_fa, FA_VERSION = _load_flash_attention()
HAS_FA = _fa is not None

# Override for testing: set to 'fa3', 'fa2', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa():
    """Decide once whether to use FA, based on availability, override, and dtype."""
    if _override_impl in ('fa3', 'fa2', 'fa'):
        assert HAS_FA, "Cannot override to FA: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA:
        from nanochat.common import COMPUTE_DTYPE
        if FA_VERSION == 'fa3':
            # FA3 Hopper kernels only support bf16 and fp8
            return COMPUTE_DTYPE == torch.bfloat16
        else:
            # FA2 supports bf16 and fp16
            return COMPUTE_DTYPE in (torch.bfloat16, torch.float16)
    return False

USE_FA = _resolve_use_fa()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _sdpa_varlen_attention(q, k, v, max_seqlen, window_size, enable_gqa):
    """
    SDPA fallback for varlen: reshapes packed (T, H, D) to (B, T_seq, H, D)
    and uses standard causal SDPA. No document isolation (cross-doc bleeding
    within each T_seq chunk), but uses efficient is_causal=True kernels.
    """
    T, H, D = q.shape
    H_kv = k.shape[1]
    B = T // max_seqlen
    q = q.view(B, max_seqlen, H, D).transpose(1, 2)
    k = k.view(B, max_seqlen, H_kv, D).transpose(1, 2)
    v = v.view(B, max_seqlen, H_kv, D).transpose(1, 2)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2).reshape(T, H, D)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k,
                           causal=False, window_size=(-1, -1)):
    """
    Flash Attention for packed variable-length sequences (training, no KV cache).

    1D packed inputs where multiple documents are concatenated into one buffer.
    Each document attends only to itself, with boundaries defined by cu_seqlens.

    Args:
        q, k, v: Tensors of shape (total_tokens, H, D)
        cu_seqlens_q, cu_seqlens_k: Cumulative sequence lengths, shape (max_num_seqs,).
            Format: [0, end_doc1, end_doc2, ..., total, total, ...]
        max_seqlen_q, max_seqlen_k: Max individual sequence length (FA3 tiling hint).
        causal: Whether to use causal masking.
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (total_tokens, H, D)
    """
    if USE_FA:
        return _fa.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            causal=causal, window_size=window_size,
        )

    # SDPA fallback: reshape to (B, T_seq) and use standard causal SDPA (no doc isolation)
    enable_gqa = q.size(1) != k.size(1)
    return _sdpa_varlen_attention(q, k, v, max_seqlen_q, window_size, enable_gqa)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA and hasattr(_fa, 'flash_attn_with_kvcache'):
        return _fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3/FA2)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_varlen_func=flash_attn_varlen_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
