#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Add uv to PATH (it installs to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# Ensure we're using the venv Python and torchrun
PYTHON=".venv/bin/python"
TORCHRUN=".venv/bin/torchrun"

# Install flash_attn if the wheel exists (for A100 compatibility)
if [ -f "flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl" ]; then
    uv pip install flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl
fi

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# You can authenticate in one of two ways:
# 1) Set WANDB_API_KEY environment variable before running:
#    `export WANDB_API_KEY=your_api_key_here`
#    `bash runs/speedrun.sh`
# 2) Or run `wandb login` after the venv is set up (the venv will be active)
#    The script will automatically use wandb if WANDB_API_KEY is set or if you've logged in.
# Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash runs/speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# If WANDB_API_KEY is set, export it so wandb can use it automatically
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
    echo "Using WANDB_API_KEY from environment for wandb authentication"
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
$PYTHON -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
$PYTHON -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 370 is the right number here
$PYTHON -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
$PYTHON -m scripts.tok_train --max-chars=20000000 --vocab-size=50304
# evaluate the tokenizer (report compression ratio etc.)
$PYTHON -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. Also, the new DataLoader wastes about 35% of tokens to cropping
# so 240 / (1 - 0.35) = 370 shards are needed.
# At ~100MB/shard, this downloads ~37GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
NPROC_PER_NODE=8
# Per-device batch size (reduce this if you hit OOM - gradient accumulation will automatically increase) Default is 32. 
# To match modded-nanogpt initial batch: 8 seqs * 2048 seq_len * 8 GPUs = 131,072 tokens
DEVICE_BATCH_SIZE=8
TOTAL_BATCH_SIZE=131072

# pretrain the d20 model
#$TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train-mine -- --depth=12 --target-param-data-ratio=20 --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
$TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train-main-profiled -- --depth=11 --target-param-data-ratio=20 --device-batch-size=$DEVICE_BATCH_SIZE --total-batch-size=$TOTAL_BATCH_SIZE --run=$WANDB_RUN
# # evaluate the model on a larger chunk of train/val data and draw some samples
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# # evaluate the model on CORE tasks
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# # -----------------------------------------------------------------------------
# # Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# # download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# # see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# # run midtraining and eval the model
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# # -----------------------------------------------------------------------------
# # Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# # train sft and re-eval right away (should see a small bump)
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
# $TORCHRUN --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# # chat with the model over CLI! Leave out the -p to chat interactively
# # python -m scripts.chat_cli -p "Why is the sky blue?"

# # even better, chat with your model over a pretty WebUI ChatGPT style
# # python -m scripts.chat_web

# # -----------------------------------------------------------------------------
# # Reinforcement Learning. Optional, and currently only on GSM8K
# # (optional)

# # run reinforcement learning
# # torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# # eval the RL model only on GSM8K
# # torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# # -----------------------------------------------------------------------------
# # Generate the full report by putting together all the sections
# # report.md is the output and will be copied to current directory for convenience
# $PYTHON -m nanochat.report generate
