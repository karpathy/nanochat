#!/bin/bash

# Nanobot: same pipeline as speedrun — Pretrain (base) → SFT (chat) → serve.
# This script runs pretrain + SFT; when it finishes you get SFT checkpoints you can serve.
# Checkpoints go under model-tag "nanobot" (base_checkpoints/nanobot, chatsft_checkpoints/nanobot).
# Designed to run on an 8XH100 GPU node; ~3 hours.

# Example: bash runs/nanobot.sh
# With wandb: WANDB_RUN=nanobot bash runs/nanobot.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=nanobot
fi

python -m nanochat.report reset

# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 \
  --run="$WANDB_RUN" --model-tag=nanobot
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# SFT (chat)
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
  --device-batch-size=16 --run="$WANDB_RUN" --model-tag=nanobot
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft --model-tag=nanobot

# Serve: python -m scripts.chat_web  (use nanobot checkpoint via env or default latest)

python -m nanochat.report generate
