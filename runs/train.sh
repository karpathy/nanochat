#!/bin/bash

# Terminate the script if any command exits with a non-zero status (error).
set -e -o pipefail

# -----------------------------------------------------------------------------
# Configuration (everything is overridable by setting these env variables)
SKIP_SETUP=${SKIP_SETUP:-} # Set to 1 to skip installing uv and dependencies
# Extra training dependencies to install (cpu or gpu)
EXTRA_DEPENDENCIES=${EXTRA_DEPENDENCIES:-gpu}
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
WANDB_RUN=${WANDB_RUN:-dummy} # by default use "dummy" : it's handled as a special case, skips logging to wandb
WRITE_REPORT=${WRITE_REPORT:-1} # Set to 0 to skip writing the markdown report
# Number of dataset shards to download in background (set to 0 to skip
# downloading). Each data shard is ~250M chars (~100MB of compressed text). At
# least 8 shards (~2B characters) are needed for tokenizer training. The maximum
# total number of shards available in the entire dataset is 1822.
DATASET_SHARDS_TO_DOWNLOAD=${DATASET_SHARDS_TO_DOWNLOAD:-370}
TRAIN_TOKENIZER=${TRAIN_TOKENIZER:-1} # Set to 0 to skip tokenizer training
# Default intermediate artifacts directory is in ~/.cache/nanochat, override by
# setting NANOCHAT_BASE_DIR to a different path.
# The number of processes (and GPUs) to use for training/eval.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_DEPTH=${MODEL_DEPTH:-26} # Number of transformer layers
# Batch size per GPU for training/eval. Decrease if you run out of memory,
# increase if you have more memory and want to go faster.
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-$DEVICE_BATCH_SIZE}
USE_FP8=${USE_FP8:-} # Set to 1 to use FP8 training, otherwise will use BF16
# Extra args to pass to base model training script, see scripts/base_train.py
EXTRA_BASE_TRAINING_ARGS=${EXTRA_BASE_TRAINING_ARGS:-}
# Extra args to pass to base model eval script, see scripts/base_eval.py
EXTRA_BASE_EVAL_ARGS=${EXTRA_BASE_EVAL_ARGS:-}
# Extra args to pass to SFT training script, see scripts/chat_sft.py
EXTRA_SFT_TRAINING_ARGS=${EXTRA_SFT_TRAINING_ARGS:-}
RUN_SFT_EVAL=${RUN_SFT_EVAL:-1} # Set to 0 to skip SFT eval
# Extra args to pass to SFT eval script, see scripts/chat_eval.py
EXTRA_SFT_EVAL_ARGS=${EXTRA_SFT_EVAL_ARGS:-}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"$HOME/.cache/nanochat"}
# Number of threads for OpenMP (set to 1 to avoid oversubscription).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# -----------------------------------------------------------------------------
# Setup
if [ -z "$SKIP_SETUP" ]; then
    mkdir -p $NANOCHAT_BASE_DIR

    # Python venv setup with uv

    # install uv (if not already installed)
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    # install the repo dependencies
    uv sync --extra $EXTRA_DEPENDENCIES
    # activate venv so that `python` uses the project's venv instead of system python
    source .venv/bin/activate
else
    source .venv/bin/activate
fi

# -----------------------------------------------------------------------------
if [ "$WRITE_REPORT" -eq "1" ]; then
    # During the course of the run, we will be writing markdown reports to the
    # report/ directory in the base dir. This command clears it out and writes a
    # header section with a bunch of system info and a timestamp that marks the
    # start of the run.
    python -m nanochat.report reset
fi

# -----------------------------------------------------------------------------
# Pretraining dataset download
if [ "$DATASET_SHARDS_TO_DOWNLOAD" -gt 0 ]; then
    # Download the first ~2B characters of pretraining dataset for tokenizer
    # training. Look at dev/repackage_data_reference.py for details on how this
    # data was prepared.
    python -m nanochat.dataset -n 8
    # Immediately also kick off downloading more shards in the background while
    # tokenizer trains (if enabled). Approximately 350 shards are needed for 10B
    # tokens of data for pretraining.
    if [ "$DATASET_SHARDS_TO_DOWNLOAD" -gt 8 ]; then
        python -m nanochat.dataset -n $DATASET_SHARDS_TO_DOWNLOAD &
        DATASET_DOWNLOAD_PID=$!
    fi
fi


# -----------------------------------------------------------------------------
# Tokenizer
if [ "$TRAIN_TOKENIZER" -eq 1 ]; then
    # train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of
    # data
    python -m scripts.tok_train --max-chars=2000000000
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Complete pretraining dataset download
if [ "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
if [ "$USE_FP8" -eq "1" ]; then
    EXTRA_BASE_TRAINING_ARGS+=" --fp8"
fi
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=$MODEL_DEPTH --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN $EXTRA_BASE_TRAINING_ARGS
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=$EVAL_BATCH_SIZE $EXTRA_BASE_EVAL_ARGS

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# Download 2.3MB of synthetic identity conversations to impart a personality to
# nanochat. See dev/gen_synthetic_data.py for details on how this data was
# prepared and to get a sense of how you can easily tune it.
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# run SFT and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN $EXTRA_SFT_TRAINING_ARGS
if [ "$RUN_SFT_EVAL" -eq "1" ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft $EXTRA_SFT_EVAL_ARGS
fi

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
if [ "$WRITE_REPORT" -eq "1" ]; then
    python -m nanochat.report generate
fi
