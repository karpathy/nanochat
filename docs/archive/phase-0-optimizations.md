---
title: "Phase 0 — Optimizations"
summary: "Completed architecture and training optimizations for nanochat."
status: archived
archived_date: "2026-03-05"
archived_reason: "All optimizations implemented and validated. 27% speedup from ClimbMix alone."
---

# Phase 0 — Optimizations ✅ COMPLETED

**Goal**: Maximize training efficiency on single-node hardware through architecture and training optimizations.

## Optimizations Implemented

### 0.1 Flash Attention 3
- **Completed**: 2026-01-11
- Replaced FA2 with FA3 via `kernels` package
- 9% improvement in tok/sec
- Simplified attention code, in-place KV cache updates, sliding window support

### 0.2 FP8 Training
- **Completed**: 2026-02-02
- Integrated torchao float8 for Linear layers
- 5-17% capability-matched speedup at d24+ scale
- 9GB memory reduction, enabled via `--fp8` flag

### 0.3 Sliding Window Attention
- **Completed**: 2026-01-11
- Configurable via `--window-pattern` (e.g., `SSSL`)
- Short window = sequence_len // 2, long = full context
- Final layer always forced to full context

### 0.4 Dataset Quality (ClimbMix)
- **Completed**: 2026-03-04
- Migrated from FineWeb-EDU 100B to ClimbMix 400B
- **27% speedup** (2h46m → 2h1m to GPT-2 capability)
- Biggest single improvement to nanochat performance

### 0.5 Optimizer Modernization (Muon)
- **Completed**: 2026-01-10 to 2026-01-22
- Polar Express orthogonalization, NorMuon variance reduction
- Cautious weight decay, weight decay schedule
- Scaling law: WD ∝ 1/width²

### 0.6 BOS-Aligned Dataloader
- **Completed**: 2026-01-13
- BestFit-Crop algorithm: 100% utilization, 34.6% crop waste
- Eliminates mid-document sequence starts

### 0.7 Per-Layer Residual Scalars
- **Completed**: 2026-01-11
- `x0_lambdas` and `resid_lambdas` per-layer scaling
- Consistent improvement across all model sizes (d8-d20)

### 0.8 Auto Batch Size Scaling
- **Completed**: 2026-02-05
- Formula: B_opt ∝ D^0.383 (Cerebras Power Lines paper)
- `--total-batch-size=-1` auto-computes from model depth

### 0.9 Explicit Dtype Management
- **Completed**: 2026-03-04
- Removed autocast, explicit COMPUTE_DTYPE
- Auto-detected per hardware, override via `NANOCHAT_DTYPE`

### 0.10 Compression-Focused Success Pattern
- **Validated**: 2026-03-05
- Successful optimizations improve compression efficiency
- Failed experiments added complexity without compression gain
- Provides theoretical framework for future decisions

## Impact

- Combined speedup: ~50%+ over original nanochat
- ClimbMix alone: 27% speedup at zero cost
- Validated compression principle as guiding framework
