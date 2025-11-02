#!/bin/bash

# Many would like to run nanochat on a a single GPU or a tiny cluster.
# This script is the "Best ChatGPT clone that a crappy single GPU can buy",
# It is designed to run in ~1 hour on a single 3080 GPU with 10GB of VRAM.
# This will help you get started. The model will be bad, terribly bad, but helps you to get started.
# Comments are sparse, see speedrun.sh for more detail.

# 1) Example launch (simplest):
# bash tinyrun.sh

# 2) Example launch in a screen session (because the run takes ~1 hour):
# screen -L -Logfile tinyrun.log -S tinyrun bash tinyrun.sh

# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=tinyrun screen -L -Logfile tinyrun.log -S tinyrun bash tinyrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
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

# Install Rust / Cargo (if not already installed)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer (if not already built)
if ! python -c "import rustbpe" &> /dev/null; then
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Download the dataset for pretraining
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi


echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Train the base model on rather smaller parameters to get a sense of the code.
# depth=the depth of the Transformer model to train
# max_seq_len=max context length
# device_batch_size=per-device batch size (set to not OOM)
# eval_tokens=number of tokens to evaluate val loss on
# core_metric_every=every how many steps to evaluate the core metric (-1 = disable)
# total_batch_size=total desired batch size, in #tokens
# num_iterations=explicit number of steps of the optimization (-1 = disable)
torchrun --standalone -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
torchrun --standalone -m scripts.base_loss --device_batch_size=1 --split_tokens=512
torchrun --standalone -m scripts.base_eval --max-per-task=16

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone -m scripts.mid_train -- --device_batch_size=1 --run=$WANDB_RUN
torchrun --standalone -m scripts.chat_eval -- -i mid

torchrun --standalone -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

python -m nanochat.report generate