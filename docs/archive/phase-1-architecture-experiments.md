---
title: "Phase 1 — Architecture Experiments"
summary: "Attempted architecture changes, all produced negative results. Validated compression principle."
status: archived
archived_date: "2026-03-05"
archived_reason: "All experiments completed with negative results. Compression analysis explains why."
---

# Phase 1 — Architecture Experiments ❌ NEGATIVE RESULTS

**Goal**: Improve model architecture through modern techniques (SwiGLU, MoE, multi-token prediction, etc.)

**Outcome**: All experiments failed. Analysis revealed they added complexity without compression benefits.

## Experiments

### 1.1 SwiGLU Activation ❌
- **Attempted**: 2026-02-05
- Replaced ReLU² with SwiGLU (3 projections, 8/3× expansion)
- **Result**: Worse on all measures (step efficiency, wall clock, FLOPs)
- **Not adopted**

### 1.2 Mixture of Experts ❌
- **Attempted**: 2026-02-19
- DeepSeekV3-style MoE: 8 routed experts, top-2 routing, 1 shared expert
- MFU dropped from ~46% to ~35% (dispatch overhead)
- FP8 support gap (`torch._grouped_mm` doesn't support FP8)
- **Not adopted**

### 1.3 Multi-Token Prediction ❌
- **Attempted**: 2026-01-12
- Predict next n tokens with weighted loss
- 13GB extra memory, worse wall-clock performance
- **Not adopted**

### 1.4 Value Embeddings ❌
- **Attempted**: 2026-01-17, reverted 2026-01-28
- Improvement tiny at d25, disappeared in wall-clock time, bloated VRAM
- **Not adopted**

### 1.5 Other Failed Experiments ❌
- **Bigram Hash Embeddings** (2026-01-27-28): Initially helped, reverted at larger scale
- **Varlen Attention** (2026-01-13): Identical performance, not worth complexity
- **Half-truncated RoPE, Asymmetric softcap, Smear gate, Backout, Skip connection** (2026-01-16): None helped

## Impact

**Compression Lens Analysis**:
- All failed experiments added architectural complexity
- None provided measurable compression efficiency gains
- Overhead (memory, compute, dispatch) exceeded theoretical benefits

**Lesson**: Architecture changes must demonstrate compression benefits at target scale before adoption. This insight directly motivated Phase 1.5 (compression-based optimization).
