---
title: "Configuration Reference"
summary: "Sectioned TOML config files, CLI overrides, and ConfigLoader usage."
read_when:
  - Setting up a training, SFT, RL, or evaluation run
  - Creating or editing a TOML config file
  - Understanding config resolution order
status: active
last_updated: "2026-06-10"
---

# Configuration Reference

Configuration is defined in `src/nanochat/common/config.py` as independent dataclasses — one per entry point. Values are loaded from a sectioned TOML file and selectively overridden via CLI flags.

## Resolution Order

Later values override earlier ones:

1. Dataclass field defaults
2. TOML file values (per section)
3. Explicit CLI flags

## TOML Config Files

Config files use TOML sections matching the entry point:

```toml
[common]
base_dir = "/path/to/experiments/nanochat"
run = "my-experiment"
wandb = "online"

[training]
depth = 12
num_iterations = 10000
device_batch_size = 32
```

Pass the file with `--config`:

```bash
uv run python -m nanochat.scripts.base_train --config path/to/config.toml
```

CLI flags override file values for that run:

```bash
uv run python -m nanochat.scripts.base_train --config config.toml --depth 6 --run quick-test
```

A copy of the resolved config is auto-saved to `checkpoints/<section>/<model_tag>/config.toml` at the start of each run.

Generate a default config with all fields and inline comments:

```bash
uv run python -m nanochat.scripts.base_train --generate-config > config.toml
```

## Base Directory

All runtime data (datasets, tokenizer, checkpoints, eval results) lives under a single root resolved in this order:

1. `base_dir` in `[common]` section of the TOML file
2. `--base-dir` CLI flag
3. `NANOCHAT_BASE_DIR` environment variable
4. Default: `~/.cache/nanochat/`

See [data-layout.md](data-layout.md) for the full directory structure.

## [common] Fields

Shared across all entry points.

| Field | Default | Description |
|-------|---------|-------------|
| `base_dir` | `null` | Override `NANOCHAT_BASE_DIR` for this run |
| `device_type` | `""` | `cuda`, `cpu`, or `mps` (empty = autodetect) |
| `run` | `"unnamed"` | WandB run name |
| `wandb` | `"local"` | WandB mode: `online`, `local`, or `disabled` |
| `wandb_project` | `"nanochat"` | WandB project name |

## [training] Fields

Base model pretraining (`nanochat.scripts.base_train`).

### Model Architecture

| Field | Default | Description |
|-------|---------|-------------|
| `depth` | `20` | Number of transformer layers |
| `aspect_ratio` | `64` | `model_dim = depth × aspect_ratio` |
| `head_dim` | `128` | Target attention head dimension |
| `max_seq_len` | `2048` | Maximum context length |
| `window_pattern` | `"SSSL"` | Sliding window pattern tiled across layers (`L`=full, `S`=half) |

### Training Horizon

Exactly one should be active (checked in order of precedence):

| Field | Default | Description |
|-------|---------|-------------|
| `num_iterations` | `-1` | Explicit number of optimizer steps |
| `target_flops` | `-1.0` | Derive iterations to reach this FLOPs budget |
| `target_param_data_ratio` | `10.5` | Derive iterations for compute-optimal D:N ratio |

### Optimization

| Field | Default | Description |
|-------|---------|-------------|
| `device_batch_size` | `32` | Per-device batch size in sequences |
| `total_batch_size` | `-1` | Total batch size in tokens (`-1` = auto) |
| `embedding_lr` | `0.3` | AdamW LR for embedding parameters |
| `unembedding_lr` | `0.008` | AdamW LR for unembedding parameters |
| `matrix_lr` | `0.02` | Muon LR for weight matrices |
| `scalar_lr` | `0.5` | LR for scalar parameters |
| `weight_decay` | `0.28` | Weight decay (auto-scaled by batch size and depth) |
| `warmup_steps` | `40` | Linear LR warmup steps |
| `warmdown_ratio` | `0.65` | Fraction of total steps for LR warmdown |
| `final_lr_frac` | `0.05` | Final LR as fraction of peak LR |
| `resume_from_step` | `-1` | Resume from checkpoint at this step (`-1` = disabled) |

### Evaluation & Checkpointing

| Field | Default | Description |
|-------|---------|-------------|
| `eval_every` | `250` | Validate bpb every N steps (`-1` = disabled) |
| `eval_tokens` | `41943040` | Tokens used for validation loss (80 × 524288) |
| `core_metric_every` | `2000` | CORE score every N steps (`-1` = disabled) |
| `core_metric_max_per_task` | `500` | Max examples per task for CORE |
| `sample_every` | `2000` | Sample from model every N steps (`-1` = disabled) |
| `save_every` | `-1` | Checkpoint every N steps (`-1` = only at end) |

### FP8 & Compression

| Field | Default | Description |
|-------|---------|-------------|
| `fp8` | `false` | Enable FP8 training (H100+ only) |
| `fp8_recipe` | `"tensorwise"` | `tensorwise` or `rowwise` |
| `track_compression` | `false` | Enable compression metrics tracking |
| `compression_log_every` | `100` | Log compression metrics every N steps |
| `track_layer_compression` | `false` | Track per-layer compression (slower) |
| `compression_early_stop` | `false` | Stop when compression plateaus |
| `model_tag` | `null` | Checkpoint directory name (default: `d<depth>`) |

## [sft] Fields

Supervised fine-tuning (`nanochat.scripts.sft_train`).

| Field | Default | Description |
|-------|---------|-------------|
| `model_tag` | `null` | Pretrained model to load (default: auto) |
| `model_step` | `null` | Checkpoint step to load (`null` = last) |
| `load_optimizer` | `true` | Resume optimizer state |
| `num_iterations` | `-1` | Steps (`-1` = full epoch) |
| `max_seq_len` | `null` | Override pretrain sequence length |
| `device_batch_size` | `null` | Override pretrain batch size |
| `total_batch_size` | `null` | Override pretrain total batch size |
| `embedding_lr` | `null` | Override pretrain embedding LR |
| `unembedding_lr` | `null` | Override pretrain unembedding LR |
| `matrix_lr` | `null` | Override pretrain matrix LR |
| `init_lr_frac` | `0.8` | Initial LR as fraction of peak |
| `warmup_ratio` | `0.0` | Warmup as fraction of total steps |
| `warmdown_ratio` | `0.5` | Warmdown as fraction of total steps |
| `final_lr_frac` | `0.0` | Final LR as fraction of peak |
| `eval_every` | `200` | Validate every N steps (`-1` = disabled) |
| `eval_tokens` | `20971520` | Tokens for validation (40 × 524288) |
| `chatcore_every` | `200` | ChatCORE eval every N steps (`-1` = disabled) |
| `chatcore_max_cat` | `-1` | Max categories (`-1` = no limit) |
| `chatcore_max_sample` | `24` | Max samples per category |
| `mmlu_epochs` | `3` | MMLU evaluation epochs |
| `gsm8k_epochs` | `4` | GSM8K evaluation epochs |

## [rl] Fields

Reinforcement learning (`nanochat.scripts.rl_train`).

| Field | Default | Description |
|-------|---------|-------------|
| `model_tag` | `null` | Model to load (default: auto) |
| `model_step` | `null` | Checkpoint step (`null` = last) |
| `num_epochs` | `1` | Training epochs |
| `device_batch_size` | `8` | Per-device batch size |
| `examples_per_step` | `16` | Examples per RL step |
| `num_samples` | `16` | Samples per example |
| `max_new_tokens` | `256` | Max tokens to generate |
| `temperature` | `1.0` | Sampling temperature |
| `top_k` | `50` | Top-k sampling |
| `embedding_lr` | `0.2` | Embedding LR |
| `unembedding_lr` | `0.004` | Unembedding LR |
| `matrix_lr` | `0.02` | Matrix LR |
| `weight_decay` | `0.0` | Weight decay |
| `init_lr_frac` | `0.05` | Initial LR fraction |
| `eval_every` | `60` | Evaluate every N steps |
| `eval_examples` | `400` | Examples for evaluation |
| `save_every` | `60` | Checkpoint every N steps |

## [evaluation] Fields

Standalone evaluation (`nanochat.scripts.evaluate`).

| Field | Default | Description |
|-------|---------|-------------|
| `modes` | `"core,bpb,sample"` | Comma-separated: `core`, `bpb`, `sample` |
| `hf_path` | `null` | HuggingFace model path (empty = use nanochat checkpoint) |
| `model_tag` | `null` | Model to evaluate (default: auto) |
| `step` | `null` | Checkpoint step (`null` = last) |
| `max_per_task` | `-1` | Max examples per task (`-1` = all) |
| `device_batch_size` | `32` | Per-device batch size |
| `split_tokens` | `20971520` | Tokens per evaluation split (40 × 524288) |
