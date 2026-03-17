"""Tests for compute dtype detection and NANOCHAT_DTYPE override."""

import torch
import pytest

import nanochat.common.dtype as dtype_mod


@pytest.fixture(autouse=True)
def reset_dtype_cache():
    """Reset module-level cache between tests."""
    dtype_mod._COMPUTE_DTYPE = None
    dtype_mod._COMPUTE_DTYPE_REASON = None
    yield
    dtype_mod._COMPUTE_DTYPE = None
    dtype_mod._COMPUTE_DTYPE_REASON = None


def test_env_override_bfloat16(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "bfloat16")
    assert dtype_mod.get_compute_dtype() == torch.bfloat16


def test_env_override_float16(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "float16")
    assert dtype_mod.get_compute_dtype() == torch.float16


def test_env_override_float32(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    assert dtype_mod.get_compute_dtype() == torch.float32


def test_reason_reflects_env(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "bfloat16")
    dtype_mod.get_compute_dtype()
    assert "NANOCHAT_DTYPE" in dtype_mod.get_compute_dtype_reason()


def test_cpu_fallback(monkeypatch):
    monkeypatch.delenv("NANOCHAT_DTYPE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert dtype_mod.get_compute_dtype() == torch.float32


def test_result_is_cached(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DTYPE", "float32")
    first = dtype_mod.get_compute_dtype()
    # Change env after first call — cached value must not change
    monkeypatch.setenv("NANOCHAT_DTYPE", "bfloat16")
    assert dtype_mod.get_compute_dtype() is first
