---
title: "Quickstart"
summary: "Get nanochat running from scratch: environment setup, data download, tokenizer training, and first training run."
read_when:
  - Setting up nanochat for the first time
  - Running on a new machine or GPU node
status: active
last_updated: "2026-06-14"
---

# Quickstart

## 1. Environment

```bash
# Install uv if not already installed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install dependencies
uv venv
uv sync --extra gpu        # GPU node
# uv sync --extra cpu      # CPU / Apple Silicon

source .venv/bin/activate
```

Set your base directory (where data, tokenizer, and checkpoints will be stored):

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

Or add it to your shell profile to persist across sessions. See [data-layout.md](../data-layout.md) for the full directory structure.

## 2. Config

Generate a config file in your project directory:

```bash
nanochat config init
```

Edit `config.toml` to set your `base_dir`, `run` name, and any training overrides. All subsequent commands will pick it up automatically via `--base-dir` or `NANOCHAT_BASE_DIR`.

See [configuration.md](../configuration.md) for all fields.

## 3. Data

Download training shards. Each shard is ~100MB of compressed text.

```bash
nanochat data download -n 8      # ~800MB, enough for tokenizer training
nanochat data download -n 170    # ~17GB, enough for GPT-2 capability pretraining
```

The last shard is always reserved as the validation split.

## 4. Tokenizer

Train the BPE tokenizer on ~2B characters of data (~30–60 minutes):

```bash
nanochat data tokenizer train
nanochat data tokenizer eval     # optional: check compression ratio
```

## 5. First Training Run

### Quick test (CPU / Apple Silicon, ~30 minutes)

```bash
nanochat train base \
    --depth=6 \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --num-iterations=500 \
    --eval-every=100 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --wandb=disabled
```

### GPU run (single GPU)

```bash
nanochat train base \
    --depth=12 \
    --num-iterations=2000 \
    --eval-every=250 \
    --wandb=disabled
```

### Multi-GPU run (8× H100, GPT-2 capability)

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- \
    --depth=24 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --fp8 \
    --run=my-run
```

See [runs/speedrun.sh](../../runs/speedrun.sh) for the full reference pipeline including SFT and evaluation.

## 6. SFT

After pretraining, download identity data and run supervised fine-tuning:

```bash
curl -L -o $NANOCHAT_BASE_DIR/identity.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

nanochat train sft --run=my-run
```

## 7. Chat

```bash
nanochat chat -p "Why is the sky blue?"   # single prompt
nanochat chat                              # interactive session
nanochat serve                             # web UI at http://localhost:8000
```

## 8. Status Check

```bash
# Check GPU
nvidia-smi

# Run tests
uv run pytest tests/ -q
```
