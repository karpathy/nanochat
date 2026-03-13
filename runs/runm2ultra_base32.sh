#!/bin/bash

# Recommended first serious base-model run for Apple M2 Ultra class machines.
# This is a single-device MPS run aimed at moving beyond the original GPT-2-scale framing.

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
BOOTSTRAP_SHARDS="${NANOCHAT_BOOTSTRAP_SHARDS:-170}"
TOKENIZER_MAX_CHARS="${NANOCHAT_TOKENIZER_MAX_CHARS:-2000000000}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra macos
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

export PYTHONPATH="$PWD"

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
TOKENIZER_PKL="$TOKENIZER_DIR/tokenizer.pkl"
TOKEN_BYTES="$TOKENIZER_DIR/token_bytes.pt"
DATA_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"

if [ ! -d "$DATA_DIR" ] || ! ls "$DATA_DIR"/*.parquet >/dev/null 2>&1; then
    echo "== Bootstrapping base dataset shards ($BOOTSTRAP_SHARDS train shards + val) =="
    python -m nanochat.dataset -n "$BOOTSTRAP_SHARDS"
fi

if [ ! -f "$TOKENIZER_PKL" ] || [ ! -f "$TOKEN_BYTES" ]; then
    echo "== Training tokenizer because tokenizer artifacts are missing =="
    python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS"
fi

python -m scripts.base_train \
    --device-type=mps \
    --depth=32 \
    --window-pattern=L \
    --max-seq-len=1024 \
    --device-batch-size=2 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --sample-every=100 \
    --core-metric-every=-1 \
    --run="$WANDB_RUN"