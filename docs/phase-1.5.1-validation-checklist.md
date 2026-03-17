---
title: "Phase 1.5.1 Validation Checklist"
summary: "Step-by-step checklist for running compression metrics validation experiments."
read_when: "Ready to validate compression metrics with actual training runs."
status: draft
last_updated: "2026-06-14"
---

# Phase 1.5.1 Validation Checklist

**Implementation**: ✅ `CompressionMetrics` class + training loop integration complete  
**Next step**: Run Experiment 1 (smoke test), then proceed in order.

## Pre-Flight

### 1. Environment

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

- CUDA → run all experiments
- MPS → limited scale only (Experiments 1–2, use small `--device-batch-size`)
- Neither → cannot run validation

### 2. Data

```bash
nanochat data download -n 8    # minimum for d12 smoke test
```

See [data-layout.md](data-layout.md) for where shards are stored.

### 3. Tokenizer

```bash
nanochat data tokenizer train
```

### 4. Code

```bash
python -c "from nanochat.compression_metrics import CompressionMetrics; print('OK')"
uv run pytest tests/ -q
# Expected: 165 passed, 10 skipped
```

## Experiments

Run in order. Each builds confidence before committing more compute.

### Experiment 1 — Smoke Test (d8, ~17 min on MPS / ~5 min on GPU)

Verify compression tracking works end-to-end without errors.

```bash
nanochat train base \
    --depth=8 \
    --num-iterations=100 \
    --track-compression \
    --compression-log-every=10 \
    --eval-every=50 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --wandb=disabled \
    --run=compression-smoke-d8
```

- [ ] No errors
- [ ] `[compression]` lines appear in console every 10 steps
- [ ] MFU similar to baseline run at same depth

### Experiment 2 — Short Validation (d12, ~6h on MPS / ~1–2h on GPU)

Collect enough data points to analyze correlation.

```bash
nanochat train base \
    --depth=12 \
    --track-compression \
    --compression-log-every=50 \
    --eval-every=250 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run=compression-validation-d12
```

- [ ] Completes without error
- [ ] Multiple val checkpoints logged (every 250 steps)
- [ ] Can plot `compression_ratio` vs `val_bpb` over time

### Experiment 3 — Medium Scale (d16, ~4–6h on GPU)

Validate correlation holds at larger scale.

```bash
# Multi-GPU
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- \
    --depth=16 \
    --track-compression \
    --compression-log-every=100 \
    --eval-every=500 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run=compression-validation-d16

# Single GPU: replace torchrun with: nanochat train base
```

- [ ] Completes without error
- [ ] Correlation pattern consistent with Experiment 2

### Experiment 4 — Full Scale (d24, ~8–12h on GPU)

Final validation at production scale.

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- \
    --depth=24 \
    --device-batch-size=16 \
    --track-compression \
    --compression-log-every=100 \
    --eval-every=500 \
    --run=compression-validation-d24
```

- [ ] Completes without error
- [ ] Correlation holds at scale

## Analysis

After experiments, compute correlation between `compression/compression_ratio` and `val/bpb`.

- Pearson R² between compression ratio and val loss
- Whether compression plateau precedes loss plateau (early stopping signal)
- Per-layer compression distribution (if `--track-layer-compression` used)

Write results to `docs/phase-1.5.1-validation-report.md`.

## Success Criteria

| R² | Result | Action |
|---|---|---|
| > 0.7 | Strong | Proceed to Phase 1.5.2 (dataset quality) |
| 0.4–0.7 | Moderate | Refine metrics, investigate which ones correlate |
| < 0.4 | Weak | Re-evaluate approach or pivot |

## Exit Criteria

- [ ] All planned experiments completed
- [ ] R² calculated and documented
- [ ] Validation report written
- [ ] Decision made: proceed / refine / pivot
