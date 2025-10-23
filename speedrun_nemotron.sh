#!/bin/bash

# This script is based on the "Best ChatGPT clone that $100 can buy" and modified for the nemotron datasets.
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun_nemotron.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun_nemotron.log -S speedrun_nemotron bash speedrun_nemotron.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun_nemotron screen -L -Logfile speedrun_nemotron.log -S speedrun_nemotron bash speedrun_nemotron.sh
set -x

PRE_TRAINING_DATA_NAME=nemotron_climbmix
POST_TRAINING_DATA_NAME=nemotron
TOKENIZER_NAME="tokenizer_${PRE_TRAINING_DATA_NAME}"

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PRE_TRAINING_DATA_DIR="$NANOCHAT_BASE_DIR/${PRE_TRAINING_DATA_NAME}"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun_nemotron.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset --exp_name=$WANDB_RUN

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# download and process the nemotron_climbmix data
python data/nemotron_climbmix/process_climbmix.py --data_dir=$PRE_TRAINING_DATA_DIR

python -m scripts.tok_train --max_chars=2000000000 --data_dir=$PRE_TRAINING_DATA_DIR --tokenizer_name=$TOKENIZER_NAME
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval --tokenizer_name=$TOKENIZER_NAME

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d20 model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN  --data_dir=$PRE_TRAINING_DATA_DIR  --tokenizer_name=$TOKENIZER_NAME --model_tag=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss  --data_dir=$PRE_TRAINING_DATA_DIR --tokenizer_name=$TOKENIZER_NAME --model_tag=$WANDB_RUN
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval --model_tag=$WANDB_RUN

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN  --model_tag=$WANDB_RUN --dataset_choice=$POST_TRAINING_DATA_NAME
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid --model_tag=$WANDB_RUN

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN --model_tag=$WANDB_RUN --dataset_choice=$POST_TRAINING_DATA_NAME
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft --model_tag=$WANDB_RUN

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate --exp_name=$WANDB_RUN
