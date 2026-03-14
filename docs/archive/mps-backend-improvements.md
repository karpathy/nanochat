---
title: "MPS Backend Improvements"
summary: "Archive of MPS backend improvements — fp16 dtype detection, torch.mps.synchronize/memory reporting, and empty_cache between eval steps."
status: archived
archived_date: "2026-03-15"
archived_reason: "Completed — all three MPS improvements implemented and benchmarked (a3d443d, 833624e)"
---

# MPS Backend Improvements

## What Was Done

Three changes to improve MPS training performance and observability, benchmarked on M3 Max
with PyTorch 2.9.1.

## Completed Tasks

- [x] **dtype.py**: Detect MPS and return `torch.float16` instead of `torch.float32` — halves memory, enables GradScaler, ~10-30% speed gain
- [x] **base_train.py / chat_sft.py**: Use `torch.mps.synchronize()` for accurate step timing and `torch.mps.current_allocated_memory()` for memory reporting
- [x] **base_train.py / chat_sft.py**: Call `torch.mps.empty_cache()` after eval steps to reclaim memory during long runs

## Additional Changes

- Replaced hardcoded dtype hack in `evaluation/engine.py` with `get_compute_dtype()`
- Extracted `get_device_sync(device_type)` into `common/hardware.py` to eliminate duplication
- Updated KV cache test to use `get_compute_dtype()` instead of hardcoded `torch.float32`
- Updated MPS guide with benchmark results: torch.compile ~17% speedup, SDPA sliding window no penalty, pin_memory/non_blocking non-issue on unified memory

## Benchmarks (M3 Max, PyTorch 2.9.1)

- fp16/bf16 matmuls: ~10-30% faster than fp32
- SDPA fp16: ~25% faster than fp32
- torch.compile on Muon-like workload: ~17% faster than eager
- SDPA sliding window (explicit mask) vs full context: identical speed (4.08ms/call)

## Artifacts

- [MPS Guide](../m3-max-guide.md)
- Commits: `a3d443d` (implementation), `833624e` (get_device_sync refactor)
