---
title: "Apple Silicon (MPS) Guide"
summary: "How nanochat detects and uses the MPS backend on Apple Silicon, including dtype behavior, known limitations, and practical training guidelines."
read_when:
  - Training or evaluating on Apple Silicon (M-series) hardware
  - Debugging MPS-specific issues or performance
  - Choosing batch size and sequence length for MPS
status: active
last_updated: "2026-03-15"
---

# Apple Silicon (MPS) Guide

Nanochat supports Apple Silicon via PyTorch's MPS backend. This guide covers how the
codebase handles MPS, what works, what doesn't, and practical guidelines for training.

## Device Detection

Device type is autodetected by `autodetect_device_type()` in `common/distributed.py`:

```
CUDA available → "cuda"
MPS available  → "mps"
otherwise      → "cpu"
```

Override with `--device-type=mps` (or `cpu`, `cuda`). When MPS is selected, `compute_init`
asserts `torch.backends.mps.is_available()` and sets `torch.manual_seed(42)`.

No `torch.set_float32_matmul_precision("high")` is applied — that's CUDA-only (TF32).

## Compute Dtype

MPS uses **float16** via `torch.amp.autocast`. The dtype detection in `common/dtype.py`
detects MPS and returns `torch.float16`, giving ~10-30% faster matmuls and halved memory
compared to fp32 (benchmarked on M3 Max, PyTorch 2.9.1).

This means:
- GradScaler is active (required for fp16 to handle overflow)
- Memory per parameter is ~2 bytes (same as bf16 on CUDA)
- Checkpoints trained on CUDA with bf16 are converted to fp32 on load (`checkpoint.py`)

Override with `NANOCHAT_DTYPE=float32` if you hit fp16 stability issues.

## Attention

Flash Attention 3 requires Hopper (sm90) hardware. On MPS, nanochat automatically falls
back to PyTorch's `scaled_dot_product_attention` (SDPA). This is handled transparently
in `flash_attention.py` — no configuration needed.

## What Works

| Feature | Status | Notes |
|---------|--------|-------|
| Base training (`base_train.py`) | ✅ | Full training loop works |
| SFT training (`chat_sft.py`) | ✅ | Full fine-tuning loop works |
| Evaluation (`chat_eval.py`) | ✅ | All eval tasks work |
| SDPA attention | ✅ | Automatic fallback from FA3 |
| Compression tracking | ✅ | `--track-compression` works |
| Checkpoint save/load | ✅ | bf16→fp32 conversion on load |
| `torch.compile` | ⚠️ | Called unconditionally — MPS support is limited in PyTorch, may silently fall back to eager |
| Muon optimizer | ✅ | Compiled kernels (`zeropower_via_newtonschulz5`, `muon_step`) may fall back to eager |
| `torch.mps.synchronize()` | ✅ | Used for accurate step timing |
| `torch.mps.empty_cache()` | ✅ | Called between eval and training steps |
| Memory reporting | ✅ | `torch.mps.current_allocated_memory()` for peak memory stats |

## What Doesn't Work

| Feature | Reason |
|---------|--------|
| FP8 training (`--fp8`) | Requires CUDA — flag is ignored with a warning |
| Flash Attention 3 | Requires Hopper GPU |
| bf16 compute | MPS supports bf16 matmuls but nanochat uses fp16 (GradScaler compatibility) |
| DDP / multi-device | MPS is single-device only |
| `pin_memory` / `non_blocking` | Gated on `use_cuda` — disabled on MPS |

## Known Workarounds in Code

**int64 comparison on MPS** (`evaluation/loss_eval.py`):
MPS lacks an int64 kernel for the `< 0` operator. The code uses `y.int() < 0` (int32 cast)
instead of comparing int64 tensors directly.

**Peak FLOPS** (`scripts/base_train.py`):
`get_peak_flops` returns `float("inf")` for non-CUDA devices, which effectively disables
MFU (model FLOPS utilization) reporting.

## Practical Guidelines

### CLI Example

```bash
python -m nanochat.scripts.base_train \
    --device-type=mps \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --total-batch-size=524288
```

### Batch Size Recommendations (128GB Unified Memory)

Training uses fp16 on MPS (~2 bytes per parameter).

| Depth | Params | `--device-batch-size` | `--max-seq-len` | Notes |
|-------|--------|-----------------------|-----------------|-------|
| 4     | ~10M   | 32                    | 2048            | Comfortable |
| 8     | ~42M   | 16                    | 2048            | Comfortable |
| 12    | ~110M  | 8                     | 1024            | Good for validation |
| 16    | ~235M  | 4                     | 1024            | Comfortable |
| 20    | ~400M  | 2                     | 1024            | Tight — reduce seq len if OOM |

Gradient accumulation via `--total-batch-size` maintains effective batch size regardless
of `--device-batch-size`.

### Memory Management

The training scripts call `torch.mps.empty_cache()` automatically between eval and training
steps to reclaim memory. Step timing uses `torch.mps.synchronize()` for accurate measurements,
and peak memory is reported via `torch.mps.current_allocated_memory()`.

### Overnight Runs

```bash
nohup python -m nanochat.scripts.base_train \
    --device-type=mps \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --run=mps-d12-overnight \
    --save-every=1000 \
    > train.log 2>&1 &
```

Use `--save-every` to checkpoint periodically in case of interruption.
Use `--run=dummy` to disable wandb logging for local experiments.
