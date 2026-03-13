#!/bin/bash

# Single-GPU training script for RTX 4070 (12GB VRAM)
# Trains a d12 model (~290M params) end-to-end: tokenizer → pretrain → SFT → chat
# Expected wall time: ~6-10 hours for pretraining, ~1 hour for SFT

# Run as:
# bash runs/run4070.sh
# With wandb logging:
# WANDB_RUN=d12-4070 bash runs/run4070.sh

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Python venv setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download initial shards for tokenizer training
python -m nanochat.dataset -n 8
# Download remaining shards in background (~40 needed for d12 at ratio 10.5)
python -m nanochat.dataset -n 50 &
DATASET_DOWNLOAD_PID=$!
# Train tokenizer
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining (single GPU, d12, ~290M params)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --max-seq-len=1024 \
    --total-batch-size=65536 \
    --eval-every=500 \
    --eval-tokens=1048576 \
    --core-metric-every=5000 \
    --sample-every=500 \
    --run=$WANDB_RUN \
    --model-tag=d12

# Evaluate base model
python -m scripts.base_eval --device-batch-size=4

# -----------------------------------------------------------------------------
# SFT

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --device-batch-size=8 \
    --max-seq-len=1024 \
    --total-batch-size=65536 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --run=$WANDB_RUN

python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo ""
echo "Done! Chat with your model:"
echo "  python -m scripts.chat_cli -p 'Why is the sky blue?'"
echo "  python -m scripts.chat_web"
