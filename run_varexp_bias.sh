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
    --depth=12 --model_dim=768 --num_heads=6 --max_seq_len=1024 \
    --expert_sizes='[[32,128],[32,640]]' --num_active_experts=4 \
    --use_bias_balancing \
    --bias_update_speed=0.001 \
    --load_balance_loss_weight=0.001 \
    --router_z_loss_weight=0.001 \
    --compute_loss_weight=0.0 \
    --device_batch_size=10 --total_batch_size=552960 --num_iterations=4521 \
    --eval_every=250 --core_metric_every=-1 --sample_every=2000 --save_every=1000
