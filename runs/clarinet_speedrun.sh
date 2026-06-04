#!/bin/bash

# Clarinet end-to-end pipeline: tokenizer + IV-conditioned pretraining + SFT + IV sweep eval.
# Designed to run on an 8xH100 node. Approximate budget: ~3.5 hours (vs ~3 for vanilla
# nanochat speedrun) — the extra is dual-source data and the IV weight sweep at the end.
#
# Example:
#   bash runs/clarinet_speedrun.sh
#   screen -L -Logfile runs/clarinet_speedrun.log -S clarinet bash runs/clarinet_speedrun.sh
#   WANDB_RUN=clarinet-v1 bash runs/clarinet_speedrun.sh

export OMP_NUM_THREADS=1
export CLARINET_BASE_DIR="${CLARINET_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$CLARINET_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
# Intended path: the `gpu` extra installs the CUDA 12.8 (Hopper/H100) torch
# build, sourced from download.pytorch.org/whl/cu128 — correct for an 8xH100
# node with open network. On environments whose network policy blocks that CDN
# (e.g. Claude Code on the web), fall back to PyPI, whose linux torch==2.9.1
# wheel is itself the +cu128 build and drives H100s identically (CUDA 12.8, NCCL).
uv sync --extra gpu || {
    echo "uv sync --extra gpu failed (PyTorch CDN unreachable?); installing from PyPI instead."
    uv pip install -e .
}
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Data: climbmix (general) + FineMath (reasoning instrument)

# Start downloading climbmix shards in the background.
# 170 climbmix shards for ~GPT-2 capacity, +20 padding.
python -m nanochat.dataset -n 8                    # 8 shards for tokenizer training
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# Prepare FineMath reasoning shards in parallel (~50 shards = ~3M docs).
python -m clarinet.prepare_reasoning_data -n 50 &
REASONING_PID=$!

# Tokenizer (must be retrained because clarinet added 3 new special tokens to SPECIAL_TOKENS).
# Trained only on general-source data so the vocab isn't biased toward math/proofs.
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model: clarinet IV-conditioned pretraining
echo "Waiting for climbmix download to complete..."
wait $DATASET_DOWNLOAD_PID
echo "Waiting for FineMath prep to complete..."
wait $REASONING_PID

# d24 model, same compute envelope as vanilla speedrun, with clarinet's source-marker
# conditioning and CFG-style dropout. Mix ratio and dropout are the v1 defaults.
torchrun --standalone --nproc_per_node=8 -m scripts.clarinet_train -- \
    --reasoning-mix-ratio=0.3 --p-uncond=0.1 \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN

# CORE / BPB eval on the base model (single-pass, no IV combine yet — vanilla eval).
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
curl -L -o "$CLARINET_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN

# Vanilla single-pass chat eval (for a reference accuracy without IV combination)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# IV guidance weight sweep — the actual clarinet experiment.
# Single-GPU because the eval engine maintains two parallel KV caches and we
# want clean per-task accuracy numbers without DDP averaging noise on small task sets.
python -m scripts.iv_eval -i sft \
    -a GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee \
    --weights 0,0.5,1.0,1.5,2.0,3.0,5.0

# -----------------------------------------------------------------------------
# Bonus: interactive chat with a chosen weight, e.g.
#   python -m scripts.clarinet_cli --iv-weight 2.0 -p "Prove that sqrt(2) is irrational."

python -m nanochat.report generate
