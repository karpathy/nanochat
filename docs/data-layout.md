---
title: "Data Directory Layout"
summary: "Structure of NANOCHAT_BASE_DIR — where nanochat stores data, tokenizers, checkpoints, and evaluation artifacts."
read_when:
  - Setting up nanochat for the first time
  - Configuring NANOCHAT_BASE_DIR or --base-dir
  - Understanding where checkpoints, data, or tokenizer files are stored
  - Debugging missing file errors related to data paths
status: active
last_updated: "2026-03-14"
---

# Data Directory Layout

All nanochat runtime data lives under a single root called the **base directory**.

## Base Directory Resolution

Resolved in this order (first match wins):

1. `--base-dir` CLI flag (available on `base_train`)
2. `NANOCHAT_BASE_DIR` environment variable
3. Default: `~/.cache/nanochat/`

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
└── identity.jsonl               # Identity data for SFT
```

## Path Management

All paths are defined in `src/nanochat/paths.py` — the single source of truth. No module constructs paths from `base_dir` directly. Available functions:

| Function | Returns |
|----------|---------|
| `data_dir()` | `data/climbmix/` |
| `tokenizer_dir()` | `tokenizer/` |
| `checkpoint_dir(phase, tag)` | `checkpoints/{base,sft,rl}/<tag>/` |
| `checkpoints_dir(phase)` | `checkpoints/{base,sft,rl}/` |
| `eval_tasks_dir()` | `data/eval_tasks/` |
| `eval_results_dir()` | `eval/` |
| `identity_data_path()` | `identity.jsonl` |

All functions accept an optional `base_dir` override.

## Model Tags

The `<model_tag>` subdirectory defaults to `d<depth>` (e.g. `d12`, `d20`). Override with `--model-tag`.

Checkpoint files use zero-padded step numbers: `model_000500.pt`, `meta_000500.json`. Optimizer state is sharded per DDP rank: `optim_000500_rank0.pt`.

## Training Data

Shards are downloaded on demand by `python -m nanochat.data.dataset -n <count>`. The last shard (`shard_06542`) is always the validation split.

## External Models

External models (e.g. HuggingFace checkpoints loaded via `--hf-path`) are managed by HuggingFace's own cache (`~/.cache/huggingface/` or `HF_HOME`). They are not stored under `NANOCHAT_BASE_DIR` — this keeps nanochat's data directory focused on artifacts it produces.

## Quick Setup

```bash
# Option A: Use default (~/.cache/nanochat/)
python -m nanochat.data.dataset -n 170
python -m nanochat.scripts.tok_train

# Option B: Use project-local directory
export NANOCHAT_BASE_DIR=$(pwd)/data
python -m nanochat.data.dataset -n 170
python -m nanochat.scripts.tok_train

# Option C: Per-run override
python -m nanochat.scripts.base_train --base-dir /data/nanochat --depth 12
```
