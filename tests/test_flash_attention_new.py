import sys
import types

import pytest
import torch

import nanochat.flash_attention as fa


def test_load_flash_attention_3_paths(monkeypatch):
    monkeypatch.setattr(fa.torch.cuda, "is_available", lambda: False)
    assert fa._load_flash_attention_3() is None

    monkeypatch.setattr(fa.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fa.torch.cuda, "get_device_capability", lambda: (8, 0))
    assert fa._load_flash_attention_3() is None

    monkeypatch.setattr(fa.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setitem(sys.modules, "kernels", None)
    assert fa._load_flash_attention_3() is None

    class K:
        @staticmethod
        def get_kernel(_name):
            return types.SimpleNamespace(flash_attn_interface="iface")

    monkeypatch.setitem(sys.modules, "kernels", K)
    assert fa._load_flash_attention_3() == "iface"


def test_use_fa3_override(monkeypatch):
    monkeypatch.setattr(fa, "HAS_FA3", False)
    monkeypatch.setattr(fa, "_override_impl", "fa3")
    with pytest.raises(AssertionError):
        fa._use_fa3()

    monkeypatch.setattr(fa, "_override_impl", "sdpa")
    assert fa._use_fa3() is False

    monkeypatch.setattr(fa, "_override_impl", None)
    monkeypatch.setattr(fa, "HAS_FA3", True)
    assert fa._use_fa3() is True


def test_sdpa_attention_branches():
    q = torch.randn(1, 2, 4, 3)
    k = torch.randn(1, 2, 4, 3)
    v = torch.randn(1, 2, 4, 3)
    y1 = fa._sdpa_attention(q, k, v, window_size=(-1, -1), enable_gqa=False)
    assert y1.shape == q.shape

    q2 = torch.randn(1, 2, 1, 3)
    k2 = torch.randn(1, 2, 6, 3)
    v2 = torch.randn(1, 2, 6, 3)
    y2 = fa._sdpa_attention(q2, k2, v2, window_size=(2, 0), enable_gqa=False)
    assert y2.shape == q2.shape

    q3 = torch.randn(1, 2, 3, 3)
    k3 = torch.randn(1, 2, 6, 3)
    v3 = torch.randn(1, 2, 6, 3)
    y3 = fa._sdpa_attention(q3, k3, v3, window_size=(3, 0), enable_gqa=False)
    assert y3.shape == q3.shape


def test_flex_attention_sliding_window_matches_sdpa(monkeypatch):
    """flex_attention sliding window path (Tq==Tk) must match explicit-mask SDPA."""
    if not fa.HAS_FLEX_ATTN:
        pytest.skip("flex_attention not available")

    torch.manual_seed(0)
    T, window = 16, 4
    q = torch.randn(1, 2, T, 8)
    k = torch.randn(1, 2, T, 8)
    v = torch.randn(1, 2, T, 8)

    # Force explicit-mask path
    monkeypatch.setattr(fa, "HAS_FLEX_ATTN", False)
    fa._block_mask_cache.clear()
    y_sdpa = fa._sdpa_attention(q, k, v, window_size=(window, 0), enable_gqa=False)

    # Force flex_attention path
    monkeypatch.setattr(fa, "HAS_FLEX_ATTN", True)
    fa._block_mask_cache.clear()
    y_flex = fa._sdpa_attention(q, k, v, window_size=(window, 0), enable_gqa=False)

    assert y_flex.shape == y_sdpa.shape
    assert torch.allclose(y_flex, y_sdpa, atol=1e-5), \
        f"flex_attention and SDPA outputs differ: max_diff={( y_flex - y_sdpa).abs().max():.6f}"


def test_public_flash_attn_paths(monkeypatch):
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)

    # FA3 path.
    class FakeFA3:
        @staticmethod
        def flash_attn_func(q, k, v, causal, window_size):
            del k, v, causal, window_size
            return q + 1

        @staticmethod
        def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False, window_size=(-1, -1)):
            del k_cache, v_cache, k, v, cache_seqlens, causal, window_size
            return q + 2

    monkeypatch.setattr(fa, "_fa3", FakeFA3())
    monkeypatch.setattr(fa, "HAS_FA3", True)
    monkeypatch.setattr(fa, "_override_impl", "fa3")
    out1 = fa.flash_attn_func(q, k, v, causal=True, window_size=(3, 0))
    assert torch.allclose(out1, q + 1)

    k_cache = torch.zeros(1, 8, 2, 4)
    v_cache = torch.zeros(1, 8, 2, 4)
    cache_seqlens = torch.zeros(1, dtype=torch.int32)
    out2 = fa.flash_attn_with_kvcache(q[:, :1], k_cache, v_cache, k=k[:, :1], v=v[:, :1], cache_seqlens=cache_seqlens, causal=True, window_size=(3, 0))
    assert torch.allclose(out2, q[:, :1] + 2)

    # SDPA path with cache insert/update.
    monkeypatch.setattr(fa, "_override_impl", "sdpa")
    out3 = fa.flash_attn_func(q, k, v, causal=True, window_size=(3, 0))
    assert out3.shape == q.shape

    out4 = fa.flash_attn_with_kvcache(
        q[:, :1],
        k_cache,
        v_cache,
        k=k[:, :1],
        v=v[:, :1],
        cache_seqlens=cache_seqlens,
        causal=True,
        window_size=(2, 0),
    )
    assert out4.shape == q[:, :1].shape

