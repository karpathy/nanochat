"""Compute dtype detection and management."""

import os

import torch

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_COMPUTE_DTYPE = None
_COMPUTE_DTYPE_REASON = None


def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return (
            torch.float32,
            f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)",
        )
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"


def _ensure_compute_dtype():
    global _COMPUTE_DTYPE, _COMPUTE_DTYPE_REASON
    if _COMPUTE_DTYPE is None:
        _COMPUTE_DTYPE, _COMPUTE_DTYPE_REASON = _detect_compute_dtype()


def get_compute_dtype():
    _ensure_compute_dtype()
    return _COMPUTE_DTYPE


def get_compute_dtype_reason():
    _ensure_compute_dtype()
    return _COMPUTE_DTYPE_REASON
