#!/bin/bash

# Apple Silicon scaling workflow for M2 Ultra class machines.
# Run as:
# bash runs/runm2ultra.sh

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
BOOTSTRAP_SHARDS="${NANOCHAT_BOOTSTRAP_SHARDS:-170}"
TOKENIZER_MAX_CHARS="${NANOCHAT_TOKENIZER_MAX_CHARS:-2000000000}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra macos
source .venv/bin/activate

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

echo "== M2 Ultra scaling benchmark =="
export PYTHONPATH="$PWD"
python dev/benchmark_mps_scaling.py \
    --depths 20,24,28,32,36 \
    --batch-sizes 1,2,4,8 \
    --seq-len 1024 \
    --steps 2 \
    --warmup-steps 1

cat <<'EOF'

Next step after reviewing the benchmark output:

python -m scripts.base_train \
    --device-type=mps \
    --depth=32 \
    --window-pattern=L \
    --max-seq-len=1024 \
    --device-batch-size=4 \
    --eval-every=100 \
    --core-metric-every=-1 \
    --sample-every=100

Adjust --depth and --device-batch-size using the benchmark table.
On Apple Silicon this is one MPS device with unified memory, not multi-GPU training.
EOF