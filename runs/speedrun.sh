#!/bin/bash

# Audience: CUDA GPU users (designed for a blank 8XH100 node). On Mac/CPU use runs/runcpu.sh instead.
# Fail-fast + idempotent: the first failed stage stops the run; re-run the same command and finished
# stages are skipped (FORCE_STAGES=all or FORCE_STAGES=tok_train,base_train to redo). See docs/RUNNING.md.

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 1.5 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~1.5 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

set -euo pipefail

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
PROFILE=speedrun
source "$(dirname "${BASH_SOURCE[0]}")/_stage_lib.sh"
# Run knobs come from nanochat/defaults.py (PROFILE_SPEEDRUN); profiles.sh is its emitted shell form.
source "$(dirname "${BASH_SOURCE[0]}")/profiles.sh"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# Python venv setup: if SKIP_SETUP is set, skip uv/venv/sync (expect .venv to exist)
if [ -z "${SKIP_SETUP:-}" ]; then
    # install uv (if not already installed)
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    # install the repo dependencies
    uv sync --extra gpu
fi
# activate venv (always happens; .venv must exist)
set +u; source .venv/bin/activate; set -u

# Hard refuse on non-CUDA: speedrun launches 8-way depth-24 training that can wedged a laptop (OOM / kernel panic).
if ! python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    log "ERROR no CUDA device detected; speedrun expects 8xH100. Use: bash runs/runcpu.sh (Mac/CPU). See docs/RUNNING.md §1/§10"
    exit 1
fi

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

log "$(pipeline_map tok:tok_train base:base_train eval:base_eval sft:chat_sft chat_eval:chat_eval)"
# speedrun uses base_train default max_seq_len (DEFAULT_SEQ_LEN=2048); profile has no override.
SHARDS_NEEDED=170
SPEEDRUN_MAX_SEQ_LEN="${SPEEDRUN_MAX_SEQ_LEN:-2048}"
print_knob_diff "$PROFILE" "$SPEEDRUN_DEPTH" "$SPEEDRUN_DEVICE_BATCH_SIZE" "$SPEEDRUN_MAX_SEQ_LEN"
print_run_tip

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python3 -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python3 -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
run_stage tok_train "" "tok_eval (or SKIP if .tok_eval_ok)" -- \
    python3 -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
run_stage tok_eval "" "base_train (or SKIP if checkpoints exist)" -- \
    python3 -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
log "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
run_stage base_train "depth=$SPEEDRUN_DEPTH device_batch_size=$SPEEDRUN_DEVICE_BATCH_SIZE" "base_eval (or SKIP if .eval_ok)" -- \
    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth="$SPEEDRUN_DEPTH" --target-param-data-ratio="$SPEEDRUN_TARGET_PARAM_DATA_RATIO" --device-batch-size="$SPEEDRUN_DEVICE_BATCH_SIZE" --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
run_stage base_eval "device_batch_size=$SPEEDRUN_EVAL_DEVICE_BATCH_SIZE" "chat_sft (or SKIP if checkpoints exist)" -- \
    torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size="$SPEEDRUN_EVAL_DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model
run_stage chat_sft "" "chat_eval (or SKIP if .eval_ok)" -- \
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
run_stage chat_eval "" "chat with your model (python3 -m scripts.chat_cli)" -- \
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

log "ALL DONE cumulative=$(fmt_dur "$SECONDS")"
write_last_run_json "$PROFILE" "$SPEEDRUN_DEPTH" "$SPEEDRUN_DEVICE_BATCH_SIZE" "$SPEEDRUN_MAX_SEQ_LEN"
echo '  try: python3 -m scripts.chat_cli -p "Why is the sky blue?"'

# chat with the model over CLI! Leave out the -p to chat interactively
# python3 -m scripts.chat_cli -p "Why is the sky blue?"
