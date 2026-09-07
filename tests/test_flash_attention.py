"""
The SDPA fallback must compute the same attention as the FA3 kernels: forward,
backward, and single-token decode against a KV cache, with and without a sliding
window. Needs a GPU with FA3 (Hopper); elsewhere SDPA is the only path and there is
nothing to compare against.

python -m pytest tests/test_flash_attention.py -v
"""

import pytest
import torch

import nanochat.flash_attention as fa
from nanochat.flash_attention import flash_attn, HAS_FA3

pytestmark = pytest.mark.skipif(not HAS_FA3, reason="needs FA3 to compare the fallback against")
DEVICE = "cuda"
DTYPE = torch.bfloat16


def with_impl(monkeypatch, impl, fn):
    """Run fn under one implementation; monkeypatch restores the module's choice afterwards."""
    monkeypatch.setattr(fa, "USE_FA3", impl == "fa3")
    return fn()


@pytest.mark.parametrize("B, T, H, KVH, D, window", [
    (2, 64, 4, 4, 32, -1),     # full causal
    (2, 128, 4, 4, 32, 32),    # sliding window
    (2, 64, 8, 2, 32, -1),     # GQA: fewer kv heads than q heads
    (4, 256, 12, 12, 64, -1),  # a model-sized layer
])
def test_forward_and_backward_agree(monkeypatch, B, T, H, KVH, D, window):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    q0 = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, generator=gen)
    k0 = torch.randn(B, T, KVH, D, device=DEVICE, dtype=DTYPE, generator=gen)
    v0 = torch.randn(B, T, KVH, D, device=DEVICE, dtype=DTYPE, generator=gen)

    def run():
        q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))
        y.sum().backward()
        return y.detach(), q.grad, k.grad, v.grad

    y_fa3, *grads_fa3 = with_impl(monkeypatch, "fa3", run)
    y_sdpa, *grads_sdpa = with_impl(monkeypatch, "sdpa", run)
    torch.testing.assert_close(y_fa3, y_sdpa, atol=1e-2, rtol=1e-2)
    for g_fa3, g_sdpa in zip(grads_fa3, grads_sdpa):
        torch.testing.assert_close(g_fa3, g_sdpa, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("window", [64, 8]) # 8 < the 32 cached tokens: the window must apply during decode too
def test_kv_cache_decode_agrees(monkeypatch, window):
    B, T_max, H, D = 2, 64, 4, 32
    T_cached = 32
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    k_init = torch.randn(B, T_cached, H, D, device=DEVICE, dtype=DTYPE, generator=gen)
    v_init = torch.randn(B, T_cached, H, D, device=DEVICE, dtype=DTYPE, generator=gen)
    q1, k1, v1 = (torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE, generator=gen) for _ in range(3))

    def run():
        k_cache = torch.zeros(B, T_max, H, D, device=DEVICE, dtype=DTYPE)
        v_cache = torch.zeros(B, T_max, H, D, device=DEVICE, dtype=DTYPE)
        k_cache[:, :T_cached] = k_init
        v_cache[:, :T_cached] = v_init
        cache_seqlens = torch.full((B,), T_cached, dtype=torch.int32, device=DEVICE)
        return flash_attn.flash_attn_with_kvcache(q1, k_cache, v_cache, k=k1, v=v1, cache_seqlens=cache_seqlens,
                                                  causal=True, window_size=(window, 0))

    y_fa3 = with_impl(monkeypatch, "fa3", run)
    y_sdpa = with_impl(monkeypatch, "sdpa", run)
    torch.testing.assert_close(y_fa3, y_sdpa, atol=1e-2, rtol=1e-2)
