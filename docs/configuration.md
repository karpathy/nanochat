---
title: "Configuration Reference"
summary: "Sectioned TOML config files, CLI overrides, and ConfigLoader usage."
read_when:
  - Setting up a training, SFT, RL, or evaluation run
  - Creating or editing a TOML config file
  - Understanding config resolution order
status: active
last_updated: "2026-06-14"
---

# Configuration Reference

Configuration is defined in `src/nanochat/config/` as independent dataclasses — one per entry point. Values are loaded from a sectioned TOML file and selectively overridden via CLI flags.

## Default Configuration

Generate a full config file with all sections and inline comments:

```bash
nanochat config init
```

This produces:

```toml
[common]
base_dir = ""              # override NANOCHAT_BASE_DIR env var (empty = use env var)
device_type = ""           # cuda | cpu | mps (empty = autodetect)
run = "unnamed"            # wandb run name
wandb = "local"            # online | local | disabled
wandb_project = "nanochat"

[training]
depth = 20
aspect_ratio = 64          # model_dim = depth * aspect_ratio
head_dim = 128
max_seq_len = 2048
window_pattern = "SSSL"    # L=full context, S=half context, tiled across layers
num_iterations = -1        # explicit step count (-1 = disabled)
target_flops = -1.0        # compute budget in FLOPs (-1 = disabled)
target_param_data_ratio = 10.5  # tokens:params ratio (Chinchilla=20)
device_batch_size = 32
total_batch_size = -1      # -1 = auto-compute optimal
embedding_lr = 0.3
unembedding_lr = 0.008
matrix_lr = 0.02
scalar_lr = 0.5
weight_decay = 0.28
warmup_steps = 40
warmdown_ratio = 0.65
final_lr_frac = 0.05
resume_from_step = -1      # -1 = disabled
eval_every = 250           # -1 = disabled
eval_tokens = 41943040       # 80 * 524288
core_metric_every = 2000   # -1 = disabled
core_metric_max_per_task = 500
sample_every = 2000        # -1 = disabled
save_every = -1            # -1 = only at end
fp8 = false
fp8_recipe = "tensorwise"  # tensorwise | rowwise
# model_tag = ""           # empty = auto (e.g. "d20")
track_compression = false
compression_log_every = 100
track_layer_compression = false
compression_early_stop = false

[sft]
# model_tag = ""           # empty = auto
# model_step = -1          # -1 = last checkpoint
load_optimizer = true
num_iterations = -1        # -1 = full epoch
max_seq_len = 2048         # inherit from pretrain if checkpoint has it
device_batch_size = 32     # inherit from pretrain if checkpoint has it
total_batch_size = 524288  # inherit from pretrain if checkpoint has it
# embedding_lr = -1.0      # -1 = inherit from pretrain
# unembedding_lr = -1.0    # -1 = inherit from pretrain
# matrix_lr = -1.0        # -1 = inherit from pretrain
init_lr_frac = 0.8
warmup_ratio = 0.0
warmdown_ratio = 0.5
final_lr_frac = 0.0
eval_every = 200
eval_tokens = 20971520       # 40 * 524288
chatcore_every = 200
chatcore_max_cat = -1      # -1 = no limit
chatcore_max_sample = 24
mmlu_epochs = 3
gsm8k_epochs = 4

[rl]
# model_tag = ""           # empty = auto
# model_step = -1          # -1 = last checkpoint
num_epochs = 1
device_batch_size = 8
examples_per_step = 16
num_samples = 16
max_new_tokens = 256
temperature = 1.0
top_k = 50
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
eval_every = 60
eval_examples = 400
save_every = 60

[evaluation]
modes = "core,bpb,sample"  # comma-separated: core | bpb | sample
# hf_path = ""             # HuggingFace model path (empty = use nanochat checkpoint)
# model_tag = ""           # empty = auto
# step = -1                # -1 = last checkpoint
max_per_task = -1          # -1 = all examples
device_batch_size = 32
split_tokens = 20971520       # 40 * 524288

[tokenizer]
vocab_size = 32768         # 2^15
max_chars = 2000000000     # 2B characters
doc_cap = 10000            # max characters per document
```

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
nanochat --config path/to/config.toml train base
```

CLI flags override file values for that run:

```bash
nanochat --config config.toml train base --depth 6 --run quick-test
```

A copy of the resolved config is auto-saved to `checkpoints/<section>/<model_tag>/config.toml` at the start of each run.

Generate a default config with all fields and inline comments:

```bash
nanochat config init
nanochat config init --output my.toml
```

## Base Directory

All runtime data (datasets, tokenizer, checkpoints, eval results) lives under a single root resolved in this order:

1. Dataclass default: `~/.cache/nanochat/`
2. `NANOCHAT_BASE_DIR` environment variable
3. `base_dir` in `[common]` section of the TOML file
4. `--base-dir` CLI flag

See [data-layout.md](data-layout.md) for the full directory structure.

## [common] Fields

Shared across all entry points.

| Field           | Default      | Description                                  |
| --------------- | ------------ | -------------------------------------------- |
| `base_dir`      | `null`       | Override `NANOCHAT_BASE_DIR` for this run    |
| `device_type`   | `""`         | `cuda`, `cpu`, or `mps` (empty = autodetect) |
| `run`           | `"unnamed"`  | WandB run name                               |
| `wandb`         | `"local"`    | WandB mode: `online`, `local`, or `disabled` |
| `wandb_project` | `"nanochat"` | WandB project name                           |

## [training] Fields

Base model pretraining (`nanochat train base`).

### Model Architecture

| Field            | Default  | Description                                                     |
| ---------------- | -------- | --------------------------------------------------------------- |
| `depth`          | `20`     | Number of transformer layers                                    |
| `aspect_ratio`   | `64`     | `model_dim = depth × aspect_ratio`                              |
| `head_dim`       | `128`    | Target attention head dimension                                 |
| `max_seq_len`    | `2048`   | Maximum context length                                          |
| `window_pattern` | `"SSSL"` | Sliding window pattern tiled across layers (`L`=full, `S`=half) |

### Training Horizon

Exactly one should be active (checked in order of precedence):

| Field                     | Default | Description                                     |
| ------------------------- | ------- | ----------------------------------------------- |
| `num_iterations`          | `-1`    | Explicit number of optimizer steps              |
| `target_flops`            | `-1.0`  | Derive iterations to reach this FLOPs budget    |
| `target_param_data_ratio` | `10.5`  | Derive iterations for compute-optimal D:N ratio |

### Optimization

| Field               | Default | Description                                           |
| ------------------- | ------- | ----------------------------------------------------- |
| `device_batch_size` | `32`    | Per-device batch size in sequences                    |
| `total_batch_size`  | `-1`    | Total batch size in tokens (`-1` = auto)              |
| `embedding_lr`      | `0.3`   | AdamW LR for embedding parameters                     |
| `unembedding_lr`    | `0.008` | AdamW LR for unembedding parameters                   |
| `matrix_lr`         | `0.02`  | Muon LR for weight matrices                           |
| `scalar_lr`         | `0.5`   | LR for scalar parameters                              |
| `weight_decay`      | `0.28`  | Weight decay (auto-scaled by batch size and depth)    |
| `warmup_steps`      | `40`    | Linear LR warmup steps                                |
| `warmdown_ratio`    | `0.65`  | Fraction of total steps for LR warmdown               |
| `final_lr_frac`     | `0.05`  | Final LR as fraction of peak LR                       |
| `resume_from_step`  | `-1`    | Resume from checkpoint at this step (`-1` = disabled) |

### Evaluation & Checkpointing

| Field                      | Default    | Description                                       |
| -------------------------- | ---------- | ------------------------------------------------- |
| `eval_every`               | `250`      | Validate bpb every N steps (`-1` = disabled)      |
| `eval_tokens`              | `41943040` | Tokens used for validation loss (80 × 524288)     |
| `core_metric_every`        | `2000`     | CORE score every N steps (`-1` = disabled)        |
| `core_metric_max_per_task` | `500`      | Max examples per task for CORE                    |
| `sample_every`             | `2000`     | Sample from model every N steps (`-1` = disabled) |
| `save_every`               | `-1`       | Checkpoint every N steps (`-1` = only at end)     |

### FP8 & Compression

| Field                     | Default        | Description                                     |
| ------------------------- | -------------- | ----------------------------------------------- |
| `fp8`                     | `false`        | Enable FP8 training (H100+ only)                |
| `fp8_recipe`              | `"tensorwise"` | `tensorwise` or `rowwise`                       |
| `track_compression`       | `false`        | Enable compression metrics tracking             |
| `compression_log_every`   | `100`          | Log compression metrics every N steps           |
| `track_layer_compression` | `false`        | Track per-layer compression (slower)            |
| `compression_early_stop`  | `false`        | Stop when compression plateaus                  |
| `model_tag`               | `null`         | Checkpoint directory name (default: `d<depth>`) |

## [sft] Fields

Supervised fine-tuning (`nanochat train sft`).

| Field                 | Default    | Description                                   |
| --------------------- | ---------- | --------------------------------------------- |
| `model_tag`           | `null`     | Pretrained model to load (default: auto)      |
| `model_step`          | `null`     | Checkpoint step to load (`null` = last)       |
| `load_optimizer`      | `true`     | Resume optimizer state                        |
| `num_iterations`      | `-1`       | Steps (`-1` = full epoch)                     |
| `max_seq_len`         | `2048`     | Override pretrain sequence length             |
| `device_batch_size`   | `32`       | Override pretrain batch size                  |
| `total_batch_size`    | `524288`   | Override pretrain total batch size            |
| `embedding_lr`        | `null`     | Override pretrain embedding LR                |
| `unembedding_lr`      | `null`     | Override pretrain unembedding LR              |
| `matrix_lr`           | `null`     | Override pretrain matrix LR                   |
| `init_lr_frac`        | `0.8`      | Initial LR as fraction of peak                |
| `warmup_ratio`        | `0.0`      | Warmup as fraction of total steps             |
| `warmdown_ratio`      | `0.5`      | Warmdown as fraction of total steps           |
| `final_lr_frac`       | `0.0`      | Final LR as fraction of peak                  |
| `eval_every`          | `200`      | Validate every N steps (`-1` = disabled)      |
| `eval_tokens`         | `20971520` | Tokens for validation (40 × 524288)           |
| `chatcore_every`      | `200`      | ChatCORE eval every N steps (`-1` = disabled) |
| `chatcore_max_cat`    | `-1`       | Max categories (`-1` = no limit)              |
| `chatcore_max_sample` | `24`       | Max samples per category                      |
| `mmlu_epochs`         | `3`        | MMLU evaluation epochs                        |
| `gsm8k_epochs`        | `4`        | GSM8K evaluation epochs                       |

## [rl] Fields

Reinforcement learning (`nanochat train rl`).

| Field               | Default | Description                     |
| ------------------- | ------- | ------------------------------- |
| `model_tag`         | `null`  | Model to load (default: auto)   |
| `model_step`        | `null`  | Checkpoint step (`null` = last) |
| `num_epochs`        | `1`     | Training epochs                 |
| `device_batch_size` | `8`     | Per-device batch size           |
| `examples_per_step` | `16`    | Examples per RL step            |
| `num_samples`       | `16`    | Samples per example             |
| `max_new_tokens`    | `256`   | Max tokens to generate          |
| `temperature`       | `1.0`   | Sampling temperature            |
| `top_k`             | `50`    | Top-k sampling                  |
| `embedding_lr`      | `0.2`   | Embedding LR                    |
| `unembedding_lr`    | `0.004` | Unembedding LR                  |
| `matrix_lr`         | `0.02`  | Matrix LR                       |
| `weight_decay`      | `0.0`   | Weight decay                    |
| `init_lr_frac`      | `0.05`  | Initial LR fraction             |
| `eval_every`        | `60`    | Evaluate every N steps          |
| `eval_examples`     | `400`   | Examples for evaluation         |
| `save_every`        | `60`    | Checkpoint every N steps        |

## [evaluation] Fields

Standalone evaluation (`nanochat eval base`).

| Field               | Default             | Description                                              |
| ------------------- | ------------------- | -------------------------------------------------------- |
| `modes`             | `"core,bpb,sample"` | Comma-separated: `core`, `bpb`, `sample`                 |
| `hf_path`           | `null`              | HuggingFace model path (empty = use nanochat checkpoint) |
| `model_tag`         | `null`              | Model to evaluate (default: auto)                        |
| `step`              | `null`              | Checkpoint step (`null` = last)                          |
| `max_per_task`      | `-1`                | Max examples per task (`-1` = all)                       |
| `device_batch_size` | `32`                | Per-device batch size                                    |
| `split_tokens`      | `20971520`          | Tokens per evaluation split (40 × 524288)                |

## [tokenizer] Fields

BPE tokenizer training (`nanochat data tokenizer train`).

| Field        | Default      | Description                     |
| ------------ | ------------ | ------------------------------- |
| `vocab_size` | `32768`      | Vocabulary size (2¹⁵)           |
| `max_chars`  | `2000000000` | Max characters to train on (2B) |
| `doc_cap`    | `10000`      | Max characters per document     |
