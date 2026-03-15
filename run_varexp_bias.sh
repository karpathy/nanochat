#!/bin/bash

# Variable-size expert experiment: DeepSeek-style bias balancing, no compute loss
# 32 small (128) + 32 big (640) experts, top-4 active, 2.0x expansion
# Sequence-level balance loss (tiny) + router z-loss + bias balancing
# usage: bash run_varexp_bias.sh

set -e
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/media/henry/MoreFiles"
mkdir -p $NANOCHAT_BASE_DIR

# venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env" 2>/dev/null || true

# Build rustbpe tokenizer if not already built
if ! python -c "import rustbpe" &> /dev/null; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# download shards
python -m nanochat.dataset -n 50

# train tokenizer if not already present
python -m scripts.tok_train --max_chars=500000000

# Variable experts: 32x128 + 32x640 (5x ratio), bias balancing, no compute loss
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=varexp-bias-32x128-32x640-top4 \
    --depth=12 --model-dim=768 --num-heads=6 --max-seq-len=1024 \
    --expert-sizes='[[32,128],[32,640]]' --num-active-experts=4 \
    --use-bias-balancing \
    --bias-update-speed=0.001 \
    --load-balance-loss-weight=0.001 \
    --router-z-loss-weight=0.001 \
    --compute-loss-weight=0.0 \
    --device-batch-size=10 --total-batch-size=552960 --num-iterations=4521 \
    --eval-every=250 --core-metric-every=-1 --sample-every=2000 --save-every=1000
