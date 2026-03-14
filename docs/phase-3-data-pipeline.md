---
title: "Phase 3: Data Pipeline"
summary: "Dataset evaluation and data mixture optimization using compression principles"
read_when: "Planning dataset evaluation, data pipeline work, or data mixture optimization"
status: draft
last_updated: 2026-03-14
---

# Phase 3: Data Pipeline

## Goal

Select and prepare optimal training data using compression-based quality metrics, systematizing the ClimbMix discovery (27% speedup over FineWeb-EDU).

## Motivation

ClimbMix delivered the biggest single improvement to nanochat — likely due to better compressibility and pattern structure. Phase 3 turns this observation into a general framework: evaluate datasets by compression quality *before* expensive training runs.

## Approach

### 3.1 Compression-Based Dataset Evaluation

Evaluate datasets by three compression properties:
- **Compression ratio** — how well the data compresses (gzip, bz2)
- **Pattern diversity** — unique n-gram patterns vs total patterns
- **Structured redundancy** — mutual information / entropy ratio

Combined quality score predicts training efficiency. Validate by training d12 models on top-ranked datasets and checking correlation with final loss.

**Candidate datasets**:

| Source | Tokens | Domain |
|--------|--------|--------|
| ClimbMix | 400B | General web + code + math |
| FineWeb-EDU | 1.3T | Educational web |
| The Stack v2 | 900B | Code |
| DCLM | 1.0T+ | Curated web |
| PeS2o | 40B | Scientific papers |
| Dolma3 | 6T | General mixture |
| Wikipedia | 4B | Encyclopedia |
| Books (Gutenberg) | 10B | Literature |

### 3.2 Dataset Mix Optimization

Find optimal mixture ratios by weighting datasets by quality score with a diversity floor to prevent over-concentration on a single source.

### 3.3 Data Quality Filtering

Filter low-quality documents using compression-based signals: minimum compression ratio, minimum word diversity, length bounds. Applied before training to improve data quality at zero training cost.

### 3.4 Tokenizer Upgrade (Optional)

Current: 32K vocab BPE. Target: 100K+ vocab with multilingual and code-aware splitting for better compression at scale.

## Dependencies

- **Phase 1.5.1**: Validates that compression metrics correlate with training performance
- **Phase 1.5.0**: Config system for reproducible experiment tracking

## Success Criteria

- Compression quality score correlates with training efficiency (R² > 0.8)
- Top-ranked dataset trains 15-30% faster than bottom-ranked
- Optimal mixture outperforms single-dataset by 10-20%
