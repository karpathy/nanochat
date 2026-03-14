---
title: "Phase 6: SP-Transformer Hybrid Architecture"
summary: "Research plan combining transformer efficiency with SP Theory advantages"
read_when: "Researching hybrid architectures, SP Theory integration, or alternatives to pure transformers"
status: draft
last_updated: 2026-03-14
---

# Phase 6: SP-Transformer Hybrid Architecture

## Goal

Combine transformer pattern matching efficiency with SP Theory's compression-based reasoning, memory consolidation, and transparency.

**Risk**: High (research, not engineering)
**Decision point**: Only pursue if Phase 1.5 shows strong compression-based improvements (>15%)

## Why Hybrid?

| | Transformer | SP Theory |
|---|---|---|
| Fast parallel processing | ✅ | ❌ (search-based) |
| Proven scaling | ✅ | ❌ (unproven at scale) |
| No catastrophic forgetting | ❌ | ✅ (additive patterns) |
| Transparent reasoning | ❌ (black box) | ✅ (pattern inspection) |
| Single-exposure learning | ❌ (data hungry) | ✅ |
| Adversarial robustness | ❌ | ✅ (global optimization) |

**Hypothesis**: Use transformer for fast implicit compression, SP alignment for explicit compression and reasoning. Combine strengths, mitigate weaknesses.

## Compression as Unifying Principle

Both architectures perform information compression, but differently:

- **Transformer**: Implicit compression in learned weights. Attention = soft pattern matching. No explicit compression objective. Opaque.
- **SP Theory**: Explicit compression via MDL principle. Multiple alignment = hard pattern matching. Compression *is* the objective. Transparent.

The hybrid uses transformer for fast feature extraction, then SP alignment for deep reasoning and memory consolidation.

## Architecture Concept

```
Input Tokens
    ↓
[Transformer Backbone]  ← Fast pattern matching (existing nanochat GPT)
    ↓
Hidden Representations
    ↓
[SP Alignment Layer]    ← Deep reasoning via multiple alignment + compression
    ↓
[Compression Memory]    ← Additive pattern storage, prevents catastrophic forgetting
    ↓
Output
```

### Key Components

- **Transformer backbone**: Existing nanochat GPT, outputs hidden states to SP layer
- **SP alignment layer**: Learnable pattern memory, cosine-similarity alignment, compression loss to encourage pattern reuse
- **Compression memory**: Stores high-compression patterns separately, preservation loss during fine-tuning prevents forgetting

### Training Strategy

1. **Stage 1**: Pretrain transformer normally (already done)
2. **Stage 2**: Freeze transformer, train SP layers only (learn alignment and memory)
3. **Stage 3**: Unfreeze all, fine-tune end-to-end with preservation loss

## Research Questions

1. Does SP alignment improve reasoning? (measure on GSM8K, MATH, ARC)
2. Does compression memory prevent catastrophic forgetting? (fine-tune on new task, measure old task retention)
3. What is the optimal transformer/SP ratio? (number of SP layers, speed vs quality)
4. Does transparency hurt performance? (interpretability overhead)
5. Can single-exposure learning work? (few-shot capability)

## Success Criteria

| Level | Criteria |
|-------|---------|
| Minimum viable | No forgetting (>90% retention), pattern inspection works, within 5% of baseline performance |
| Strong | >15% reasoning improvement, full transparency with low overhead, better few-shot |
| Transformative | >30% reasoning gains, single-exposure learning, new SOTA on multiple benchmarks |

## Decision Point

This phase is only worth pursuing if:
- Phase 1.5 compression metrics correlate strongly with performance (R² > 0.7)
- Phase 1.5.3 compression-aware training shows >15% improvement
- The theoretical foundation is validated empirically

If Phase 1.5 shows <5% improvement → pursue this phase as the alternative path (pivot from incremental compression to architectural innovation).
