---
title: "Data Directory Layout"
summary: "Structure of NANOCHAT_BASE_DIR — where nanochat stores data, tokenizers, checkpoints, and evaluation artifacts."
read_when:
  - Setting up nanochat for the first time
  - Configuring NANOCHAT_BASE_DIR or --base-dir
  - Understanding where checkpoints, data, or tokenizer files are stored
  - Debugging missing file errors related to data paths
status: active
last_updated: "2026-06-14"
---

# Data Directory Layout

All nanochat runtime data lives under a single root called the **base directory**.

## Base Directory Resolution

Resolved in this order (later overrides earlier):

1. Default: `~/.cache/nanochat/`
2. `NANOCHAT_BASE_DIR` environment variable
3. `base_dir` in `[common]` section of `config.toml`
4. `--base-dir` CLI flag

See [configuration.md](configuration.md) for the full config resolution order and TOML format.

To keep data project-local, point `NANOCHAT_BASE_DIR` at any directory (e.g. `export NANOCHAT_BASE_DIR=$(pwd)/data`).

## Directory Structure

```
$NANOCHAT_BASE_DIR/
├── data/                        # All input data
│   ├── climbmix/                # Pre-training shards (ClimbMix-400B)
│   │   ├── shard_00000.parquet
│   │   ├── ...
│   │   └── shard_06542.parquet  # Last shard = validation split
│   └── eval_tasks/              # CORE evaluation task data (auto-downloaded)
│
├── tokenizer/                   # Trained BPE tokenizer
│   ├── tokenizer.pkl            # tiktoken Encoding (pickle)
│   └── token_bytes.pt           # Per-token byte lengths (for bpb calculation)
│
├── checkpoints/                 # All model checkpoints
│   ├── base/                    # Pre-training
│   │   └── <model_tag>/         # e.g. d12, d24, or custom --model-tag
│   │       ├── config.toml
│   │       ├── model_000000.pt
│   │       ├── optim_000000_rank0.pt
│   │       └── meta_000000.json
│   ├── sft/                     # Supervised fine-tuning
│   │   └── <model_tag>/
│   └── rl/                      # Reinforcement learning
│       └── <model_tag>/
│
├── eval/                        # Evaluation results
│   └── <model_slug>.csv
│
├── report/                      # Training and evaluation reports
│
├── runs/                        # Local wandb JSONL logs (wandb = "local")
│   └── nanochat/
│       └── <run>/
│           └── wandb.jsonl
│
└── identity.jsonl               # Identity data for SFT
```

## Path Management

All paths are defined in `src/nanochat/common/paths.py` — the single source of truth. No module constructs paths from `base_dir` directly. Available functions:

| Function                               | Returns                            |
| -------------------------------------- | ---------------------------------- |
| `data_dir(base_dir)`                   | `data/climbmix/`                   |
| `legacy_data_dir(base_dir)`            | `data/fineweb/`                    |
| `tokenizer_dir(base_dir)`              | `tokenizer/`                       |
| `checkpoint_dir(base_dir, phase, tag)` | `checkpoints/{base,sft,rl}/<tag>/` |
| `checkpoint_dir(base_dir, phase)`      | `checkpoints/{base,sft,rl}/`       |
| `eval_tasks_dir(base_dir)`             | `data/eval_tasks/`                 |
| `eval_results_dir(base_dir)`           | `eval/`                            |
| `report_dir(base_dir)`                 | `report/`                          |
| `identity_data_path(base_dir)`         | `identity.jsonl`                   |

All functions create the directory if absent (except `legacy_data_dir` and `identity_data_path`).

## Model Tags

The `<model_tag>` subdirectory defaults to `d<depth>` (e.g. `d12`, `d20`). Override with `--model-tag`.

Checkpoint files use zero-padded step numbers: `model_000500.pt`, `meta_000500.json`. Optimizer state is sharded per DDP rank: `optim_000500_rank0.pt`.

## Training Data

Shards are downloaded on demand by `nanochat data download -n <count>`. The last shard (`shard_06542`) is always the validation split.

## External Models

External models (e.g. HuggingFace checkpoints loaded via `--hf-path`) are managed by HuggingFace's own cache (`~/.cache/huggingface/` or `HF_HOME`). They are not stored under `NANOCHAT_BASE_DIR` — this keeps nanochat's data directory focused on artifacts it produces.

## Quick Setup

```bash
# Option A: Use default (~/.cache/nanochat/)
nanochat data download -n 170
nanochat data tokenizer train

# Option B: Use project-local directory
export NANOCHAT_BASE_DIR=$(pwd)/data
nanochat data download -n 170
nanochat data tokenizer train

# Option C: Per-run override
nanochat --base-dir /data/nanochat train base --depth 12
```
