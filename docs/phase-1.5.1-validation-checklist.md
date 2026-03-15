---
title: "Phase 1.5.1 Validation Checklist"
summary: "Step-by-step checklist for running compression metrics validation experiments"
read_when: "Ready to validate compression metrics implementation with actual training runs"
status: draft
last_updated: 2026-03-14
---

# Phase 1.5.1 Validation Checklist

## Status

**Implementation**: ✅ CompressionMetrics class + training loop integration complete
**Next Step**: Run validation experiments to test correlation between compression metrics and model performance

## Pre-Flight Checklist

### 1. Environment

```bash
# Verify GPU
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# CUDA → full experiments
# MPS → limited scale (use --device-batch-size=1 or small)
# Neither → cannot run validation
```

**Result**: MPS ✅ — Apple Silicon (M3 Max). CUDA not available. Running at limited scale with `device_type = "mps"` in config.

### 2. Data

Data lives in `$NANOCHAT_BASE_DIR/data/climbmix/` (default: `~/.cache/nanochat/data/climbmix/`). See [data-layout.md](data-layout.md) for full directory structure.

```bash
# Check if data exists
ls /Users/geronimo/build/sp_theory/experiments/nanochat/data/climbmix/shard_00000.parquet 2>/dev/null || echo "❌ No data"

# Download (8 shards minimum for d12 quick test)
uv run python -m nanochat.data.dataset -n 8 --base-dir /Users/geronimo/build/sp_theory/experiments/nanochat
```

**Result**: ✅ 9/9 shards downloaded (8 train + validation shard `shard_06542`) to `experiments/nanochat/data/climbmix/`.

Filenames are `shard_NNNNN.parquet`. Last shard is always validation.

### 3. Tokenizer

```bash
# Check if tokenizer exists
ls /Users/geronimo/build/sp_theory/experiments/nanochat/tokenizer/tokenizer.pkl 2>/dev/null || echo "❌ No tokenizer"

# Train if missing
uv run python -m nanochat.scripts.tok_train --base-dir /Users/geronimo/build/sp_theory/experiments/nanochat
```

**Result**: ✅ Trained in 43s. Saved `tokenizer.pkl` and `token_bytes.pt` to `experiments/nanochat/tokenizer/`.

### 4. Code

```bash
# Verify compression metrics available
uv run python -c "from nanochat.compression_metrics import CompressionMetrics; print('✅ OK')"

# Verify flags exist
uv run python -m nanochat.scripts.base_train --help | grep compression

# Run tests
uv run pytest tests/ -v
# Expected: 56 passed, 10 skipped
```

**Result**: ✅ `CompressionMetrics` imports cleanly. ✅ Compression flags present. ✅ 56 passed, 10 skipped. Also fixed missing `return parser` in `build_parser()`.

## Validation Experiments

Run in order. Each builds confidence before committing more compute.

### Experiment 1: Smoke Test (d8, ~10 min)

Verify compression tracking works without errors.

```bash
WANDB_MODE=disabled uv run python -m nanochat.scripts.base_train \
    --config /Users/geronimo/build/sp_theory/experiments/nanochat/nanochat.toml \
    --depth=8 --num-iterations=100 \
    --track-compression --compression-log-every=10 \
    --eval-every=50 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 \
    --run=compression-smoke-d8
```

**Check**:
- [ ] No errors
- [ ] Compression metrics in console every 10 steps
- [ ] WandB shows `compression/` metrics (entropy, compression_ratio, gzip_compression, compression_efficiency)
- [ ] MFU similar to baseline (no significant slowdown)

### Experiment 2: Short Validation (d12, ~1-2 hours)

Collect enough data to analyze correlation.

```bash
uv run python -m nanochat.scripts.base_train \
    --config /Users/geronimo/build/sp_theory/experiments/nanochat/nanochat.toml \
    --num-iterations=2000 \
    --track-compression --compression-log-every=50 \
    --eval-every=250 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 \
    --run=compression-validation-d12
```

**Check**:
- [ ] Completes 2000 iterations
- [ ] Multiple val checkpoints (every 250 steps)
- [ ] Can plot compression_ratio vs val_bpb over time

### Experiment 3: Medium Scale (d16, ~4-6 hours)

Validate at larger scale. Use multi-GPU if available.

```bash
# Multi-GPU
torchrun --nproc_per_node=8 -m nanochat.scripts.base_train \
    --config /Users/geronimo/build/sp_theory/experiments/nanochat/nanochat.toml \
    --depth=16 --num-iterations=5000 \
    --track-compression --compression-log-every=100 \
    --eval-every=500 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 \
    --run=compression-validation-d16

# Single GPU: same command with `uv run python -m` instead of torchrun
```

### Experiment 4: Full Scale (d24, ~8-12 hours)

Final validation at production scale.

```bash
torchrun --nproc_per_node=8 -m nanochat.scripts.base_train \
    --config /Users/geronimo/build/sp_theory/experiments/nanochat/nanochat.toml \
    --depth=24 \
    --track-compression --compression-log-every=100 \
    --eval-every=500 \
    --run=compression-validation-d24
```

## Analysis

After experiments, compute correlation between `compression/compression_ratio` and `val/bpb` from WandB data.

**Key metrics to compute**:
- Pearson R² between compression ratio and val loss
- Whether compression plateau precedes loss plateau (early stopping signal)
- Per-layer compression distribution (if `--track-layer-compression` used)

**Create**: `docs/phase-1.5.1-validation-report.md` with results, plots, and decision.

## Success Criteria

| Result | R² | Action |
|--------|-----|--------|
| Strong | > 0.7 | Proceed to Phase 1.5.2 (dataset evaluation) |
| Moderate | 0.4 – 0.7 | Refine metrics, investigate which ones work |
| Weak | < 0.4 | Re-evaluate approach or pivot |

## Exit Criteria

Phase 1.5.1 is complete when:

1. [ ] All experiments run successfully
2. [ ] Correlation analysis completed (R² calculated)
3. [ ] Validation report written
4. [ ] Decision made: proceed / refine / pivot
