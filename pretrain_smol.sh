#!/bin/bash

# smol pretrain script for SYNTH dataset experiments
# usage: bash pretrain_smol.sh

set -e  # exit on error
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

# download shards (~55M tokens each, need ~46 for 2.5B)
python -m nanochat.dataset -n 50

# train tokenizer if not already present
python -m scripts.tok_train --max_chars=500000000

# Model: depth=12, dim=768, 12 heads (~167M active params with MoE)
# Training: 2.5B tokens on 3 GPUs
# batch: 10 * 1024 * 3 * 18 (grad_accum) = 552,960 tokens/step
# iterations: 2.5B / 552,960 = 4,521
# batch size 9 is cursed but there are good reasons
torchrun --standalone --nproc_per_node=3 -m scripts.base_train -- \
    --run=synth-moe-12L-768d-2.5B \
    --depth=12 \
    --model_dim=768 \
    --num_heads=12 \
    --max_seq_len=1024 \
    --device_batch_size=10 \
    --total_batch_size=552960 \
    --num_iterations=4521 \
    --eval_every=250 \
    --core_metric_every=-1 \
    --sample_every=2000 \
    --save_every=1000
