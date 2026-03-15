---
title: "Configuration Reference"
summary: "TrainingConfig fields, TOML config files, and base directory resolution."
read_when:
  - Setting up a training run
  - Creating or editing a TOML config file
  - Understanding what each TrainingConfig field does
status: active
last_updated: "2026-03-16"
---

# Configuration Reference

Training is configured via `TrainingConfig` — a dataclass defined in `src/nanochat/models/config.py`. Values can be set in a TOML file and selectively overridden per-run via CLI flags.

## TOML Config Files

Save any subset of `TrainingConfig` fields to a `.toml` file:

```toml
depth = 12
base_dir = "/path/to/experiments/nanochat"
run = "my-experiment"
track_compression = true
```

Load it with `--config`:

```bash
uv run python -m nanochat.scripts.base_train --config path/to/nanochat.toml
```

CLI flags override file values — the file sets the baseline, flags patch on top.

A copy of the resolved config is auto-saved to `checkpoints/base/<model_tag>/config.toml` at the start of each run.

## Base Directory

All runtime data (data, tokenizer, checkpoints, eval results) lives under a single root resolved in this order:

1. `base_dir` field in the TOML config
2. `NANOCHAT_BASE_DIR` environment variable
3. Default: `~/.cache/nanochat/`

See [data-layout.md](data-layout.md) for the full directory structure.

## TrainingConfig Fields

### Model Architecture

| Field | Default | Description |
|-------|---------|-------------|
| `depth` | *(required)* | Number of transformer layers |
| `aspect_ratio` | `64` | `model_dim = depth × aspect_ratio` |
| `head_dim` | `128` | Target attention head dimension |
| `max_seq_len` | `2048` | Maximum context length |
| `window_pattern` | `"SSSL"` | Sliding window pattern tiled across layers (`L`=full context, `S`=half context) |

### Training Horizon

Exactly one of these should be active (checked in order of precedence):

| Field | Default | Description |
|-------|---------|-------------|
| `num_iterations` | `-1` | Explicit number of optimizer steps |
| `target_flops` | `-1.0` | Derive iterations to reach this FLOPs budget |
| `target_param_data_ratio` | `10.5` | Derive iterations for compute-optimal D:N ratio |

### Optimization

| Field | Default | Description |
|-------|---------|-------------|
| `device_batch_size` | `32` | Per-device batch size in sequences |
| `total_batch_size` | `-1` | Total batch size in tokens (`-1` = auto-compute optimal) |
| `embedding_lr` | `0.3` | AdamW LR for embedding parameters |
| `unembedding_lr` | `0.008` | AdamW LR for unembedding parameters |
| `matrix_lr` | `0.02` | Muon LR for matrix parameters |
| `scalar_lr` | `0.5` | LR for scalar parameters (`resid_lambdas`, `x0_lambdas`) |
| `weight_decay` | `0.28` | Cautious weight decay for Muon (scaled automatically by batch size and depth) |
| `warmup_steps` | `40` | Steps for LR linear warmup |
| `warmdown_ratio` | `0.65` | Fraction of total iterations for LR warmdown |
| `final_lr_frac` | `0.05` | Final LR as fraction of peak LR |
| `resume_from_step` | `-1` | Resume training from this checkpoint step (`-1` = disabled) |

### Evaluation

| Field | Default | Description |
|-------|---------|-------------|
| `eval_every` | `250` | Evaluate validation bpb every N steps (`-1` = disabled) |
| `eval_tokens` | `41943040` | Number of tokens used for validation loss evaluation |
| `core_metric_every` | `2000` | Evaluate CORE metric every N steps (`-1` = disabled) |
| `core_metric_max_per_task` | `500` | Max examples per task for CORE evaluation |
| `sample_every` | `2000` | Sample from model every N steps (`-1` = disabled) |
| `save_every` | `-1` | Save checkpoints every N steps (`-1` = only at end) |

### Runtime

| Field | Default | Description |
|-------|---------|-------------|
| `device_type` | `""` | `cuda`, `cpu`, or `mps` (empty = autodetect) |
| `fp8` | `false` | Enable FP8 training (H100+ only, requires `torchao`) |
| `fp8_recipe` | `"tensorwise"` | FP8 scaling recipe: `tensorwise` or `rowwise` |

### Compression Metrics

| Field | Default | Description |
|-------|---------|-------------|
| `track_compression` | `false` | Enable compression metrics tracking |
| `compression_log_every` | `100` | Log compression metrics every N steps |
| `track_layer_compression` | `false` | Track per-layer compression (slower) |
| `compression_early_stop` | `false` | Stop training when compression plateaus |

### Output

| Field | Default | Description |
|-------|---------|-------------|
| `model_tag` | `null` | Checkpoint directory name (default: `d<depth>`) |
| `run` | `"dummy"` | WandB run name (`"dummy"` disables WandB logging) |
| `base_dir` | `null` | Override `NANOCHAT_BASE_DIR` for this run |

## Full Config Example

All fields with their defaults and inline comments:

```toml
# ── Paths ─────────────────────────────────────────────────────────────────────
base_dir = "/path/to/experiments/nanochat"  # omit to use NANOCHAT_BASE_DIR or ~/.cache/nanochat/

# ── Output ────────────────────────────────────────────────────────────────────
run = "dummy"          # wandb run name; "dummy" disables wandb
model_tag = "d12"      # checkpoint subdir name; omit to default to d<depth>

# ── Model Architecture ────────────────────────────────────────────────────────
depth = 12             # number of transformer layers (required)
aspect_ratio = 64      # model_dim = depth × aspect_ratio
head_dim = 128         # target attention head dimension
max_seq_len = 2048     # maximum context length
window_pattern = "SSSL" # L=full context, S=half context, tiled across layers

# ── Training Horizon (first active wins) ──────────────────────────────────────
num_iterations = -1              # explicit step count (-1 = disabled)
target_flops = -1.0              # derive steps from FLOPs budget (-1 = disabled)
target_param_data_ratio = 10.5   # compute-optimal D:N ratio (Chinchilla=20)

# ── Optimization ──────────────────────────────────────────────────────────────
device_batch_size = 32     # per-device batch size in sequences; reduce if OOM
total_batch_size = -1      # total tokens per step (-1 = auto-compute optimal)
embedding_lr = 0.3         # AdamW LR for embeddings
unembedding_lr = 0.008     # AdamW LR for unembedding
matrix_lr = 0.02           # Muon LR for weight matrices
scalar_lr = 0.5            # LR for resid_lambdas / x0_lambdas
weight_decay = 0.28        # Muon weight decay (auto-scaled by batch size and depth)
warmup_steps = 40          # linear LR warmup steps
warmdown_ratio = 0.65      # fraction of total steps for LR warmdown
final_lr_frac = 0.05       # final LR as fraction of peak LR
resume_from_step = -1      # resume from checkpoint at this step (-1 = disabled)

# ── Evaluation ────────────────────────────────────────────────────────────────
eval_every = 250                  # validate bpb every N steps (-1 = disabled)
eval_tokens = 41943040            # tokens used for validation loss (80 × 524288)
core_metric_every = 2000          # CORE score every N steps (-1 = disabled)
core_metric_max_per_task = 500    # max examples per task for CORE
sample_every = 2000               # sample from model every N steps (-1 = disabled)
save_every = -1                   # checkpoint every N steps (-1 = only at end)

# ── Runtime ───────────────────────────────────────────────────────────────────
device_type = ""           # "cuda" | "cpu" | "mps" | "" (autodetect)
fp8 = false                # FP8 training (H100+ only, requires torchao)
fp8_recipe = "tensorwise"  # "tensorwise" (faster) or "rowwise" (more accurate)

# ── Compression Metrics ───────────────────────────────────────────────────────
track_compression = false       # enable compression tracking
compression_log_every = 100     # log compression metrics every N steps
track_layer_compression = false # per-layer compression (slower)
compression_early_stop = false  # stop when compression plateaus
```
