#!/bin/bash
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#                                                                           #
#                     CPU Demonstration and Test Run                        #
#                                                                           #
#_-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*#
#--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*#
# This script provides an example run for exercising the project's code paths
# on a CPU or MPS (for Apple Silicon Macs).
#
# To run this script:
# bash dev/runcpu.sh
#
# IMPORTANT NOTE:
# Training Large Language Models (LLMs) is computationally intensive and requires
# significant GPU resources and budget. This script is intended as an educational
# tool and a demonstration. It allows you to verify that the code runs, but it
# will not produce a high-quality model. It is intentionally placed in the `dev/`
# directory to emphasize its development and testing purpose.

# --- Environment Setup ---
# Set the number of threads for OpenMP to 1 to avoid potential conflicts.
export OMP_NUM_THREADS=1
# Define and create the base directory for nanochat data and caches.
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Install 'uv', a fast Python package installer, if it's not already present.
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# Create a virtual environment if it doesn't exist.
[ -d ".venv" ] || uv venv
# Sync dependencies, including the 'cpu' extra for CPU-only environments.
uv sync --extra cpu
# Activate the virtual environment.
source .venv/bin/activate

# Set a dummy Weights & Biases run name if not already defined.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Install Rust if it's not already installed.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add Rust's package manager, Cargo, to the PATH.
source "$HOME/.cargo/env"
# Build the Rust-based BPE tokenizer.
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# --- Training and Evaluation Pipeline ---
# Reset any previous reports to start fresh.
python -m nanochat.report reset

# --- Tokenizer Training ---
# Download and prepare the dataset for tokenizer training.
python -m nanochat.dataset -n 4
# Train the tokenizer on approximately 1 billion characters of text.
python -m scripts.tok_train --max_chars=1000000000
# Evaluate the trained tokenizer.
python -m scripts.tok_eval

# --- Base Model Training ---
# Train a very small, 4-layer model on the CPU.
# Note: This is a minimal run for demonstration purposes.
# - Each optimization step processes a single sequence of 1024 tokens.
# - The run consists of only 50 optimization steps.
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50
# Evaluate the loss of the base model.
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
# Perform a more comprehensive evaluation of the base model.
python -m scripts.base_eval --max-per-task=16

# --- Mid-training ---
# Continue training the model on a mixture of tasks.
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100
# Evaluate the mid-trained model. Results are expected to be poor due to the minimal training.
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# --- Supervised Fine-Tuning (SFT) ---
# Fine-tune the model on a dataset of conversations.
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# --- Interactive Chat (Optional) ---
# Uncomment the following lines to interact with the model via the command line or a web interface.
#
# # Command-Line Interface
# python -m scripts.chat_cli -p "Why is the sky blue?"
#
# # Web-based Interface
# python -m scripts.chat_web

# --- Reporting ---
# Generate a final report summarizing the run.
python -m nanochat.report generate
