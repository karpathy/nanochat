#!/bin/bash

# Audience: Mac / CPU users exercising the full pipeline (tokenizer -> base -> SFT) as a small demo.
# Fail-fast + idempotent: the first failed stage stops the run; re-run the same command and finished
# stages are skipped (FORCE_STAGES=all or FORCE_STAGES=tok_train,base_train to redo). See docs/RUNNING.md.

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

set -euo pipefail

# all the setup stuff
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
PROFILE=runcpu
source "$(dirname "${BASH_SOURCE[0]}")/_stage_lib.sh"
# Run knobs come from nanochat/defaults.py (PROFILE_RUNCPU); profiles.sh is its emitted shell form.
source "$(dirname "${BASH_SOURCE[0]}")/profiles.sh"
# Modal-only overrides: Modal sets NANOCHAT_RUNCPU_* on its CPU machine. Unset (local Mac/CPU)
# keeps the profile values above, so local runs are unchanged.
RUNCPU_DEVICE_BATCH_SIZE="${NANOCHAT_RUNCPU_DEVICE_BATCH_SIZE:-$RUNCPU_DEVICE_BATCH_SIZE}"
RUNCPU_NUM_ITERATIONS="${NANOCHAT_RUNCPU_NUM_ITERATIONS:-$RUNCPU_NUM_ITERATIONS}"
RUNCPU_EVAL_DEVICE_BATCH_SIZE="${NANOCHAT_RUNCPU_EVAL_DEVICE_BATCH_SIZE:-$RUNCPU_EVAL_DEVICE_BATCH_SIZE}"

# Python venv setup: if SKIP_SETUP is set, skip uv/venv/sync (expect .venv to exist, e.g. the Modal image)
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra cpu
fi
set +u; source .venv/bin/activate; set -u
# Run record: one folder per run (run.json, milestone events, spans). Modal sets these itself.
export NANOCHAT_RUN_ID="${NANOCHAT_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-cpu}"
export NANOCHAT_RUN_cpu="${NANOCHAT_RUN_cpu:-local-cpu}"
export NANOCHAT_RUN_DIR="${NANOCHAT_RUN_DIR:-$NANOCHAT_BASE_DIR/runs/$NANOCHAT_RUN_ID}"
python3 -m nanochat.run_record start "$NANOCHAT_RUN_DIR" --mode "$NANOCHAT_RUN_cpu" || echo "WARN: run record not started" >&2
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

log "$(pipeline_map tok:tok_train base:base_train eval:base_eval sft:chat_sft)"
SHARDS_NEEDED=8
print_knob_diff "$PROFILE" "$RUNCPU_DEPTH" "$RUNCPU_DEVICE_BATCH_SIZE" "$RUNCPU_MAX_SEQ_LEN"
print_run_tip

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
run_stage tok_train "" "tok_eval (or SKIP if .tok_eval_ok)" -- \
    python -m scripts.tok_train --max-chars=2000000000
run_stage tok_eval "" "base_train (or SKIP if checkpoints exist)" -- \
    python -m scripts.tok_eval

# train a small 6 layer model
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
_SAVE_EVERY_SECS="${NANOCHAT_SAVE_EVERY_SECS:-300}"
_RESUME_FLAG="$(resume_flag base_train "${NANOCHAT_BASE_DIR}/base_checkpoints/d${RUNCPU_DEPTH}")"
run_stage base_train "depth=$RUNCPU_DEPTH device_batch_size=$RUNCPU_DEVICE_BATCH_SIZE" "base_eval (or SKIP if .eval_ok)" -- \
    python -m scripts.base_train \
    --depth="$RUNCPU_DEPTH" \
    --head-dim="$RUNCPU_HEAD_DIM" \
    --window-pattern="$RUNCPU_WINDOW_PATTERN" \
    --max-seq-len="$RUNCPU_MAX_SEQ_LEN" \
    --device-batch-size="$RUNCPU_DEVICE_BATCH_SIZE" \
    --total-batch-size="$RUNCPU_TOTAL_BATCH_SIZE" \
    --eval-every="$RUNCPU_EVAL_EVERY" \
    --eval-tokens="$RUNCPU_EVAL_TOKENS" \
    --core-metric-every="$RUNCPU_CORE_METRIC_EVERY" \
    --sample-every="$RUNCPU_SAMPLE_EVERY" \
    --num-iterations="$RUNCPU_NUM_ITERATIONS" \
    --save-every-secs="$_SAVE_EVERY_SECS" \
    $_RESUME_FLAG \
    --run=$WANDB_RUN
run_stage base_eval "device_batch_size=$RUNCPU_EVAL_DEVICE_BATCH_SIZE" "chat_sft (or SKIP if checkpoints exist)" -- \
    python -m scripts.base_eval --device-batch-size="$RUNCPU_EVAL_DEVICE_BATCH_SIZE" --split-tokens=16384 --max-per-task=16

_SFT_RESUME_FLAG="$(resume_flag chat_sft "${NANOCHAT_BASE_DIR}/chatsft_checkpoints/d${RUNCPU_DEPTH}")"
# SFT (~10 minutes on my MacBook Pro M3 Max)
run_stage chat_sft "" "chat with your model (python -m scripts.chat_cli)" -- \
    python -m scripts.chat_sft \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --save-every-secs="$_SAVE_EVERY_SECS" \
    $_SFT_RESUME_FLAG \
    --run=$WANDB_RUN

log "ALL DONE cumulative=$(fmt_dur "$SECONDS")"
write_last_run_json "$PROFILE" "$RUNCPU_DEPTH" "$RUNCPU_DEVICE_BATCH_SIZE" "$RUNCPU_MAX_SEQ_LEN"
echo '  try: python -m scripts.chat_cli -p "What is the capital of France?"'

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"
