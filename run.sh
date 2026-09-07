#!/bin/bash
# -e: stop on any stage failure. pipefail: a failure inside `python ... | tee` must
# surface as the pipeline's exit code (otherwise tee's success would mask it).
set -e -o pipefail

# The nanochat master script: runs one experiment, end to end.
#
# An experiment = (name, git commit, depth ladder, dataset). Everything derives from
# --depth, so there is no config file: the code is the config.
# See experiment_refactor.md for the full design. The pipeline:
#   setup -> experiment identity (meta.json) -> prepare (data download, tokenizer, packing)
#   -> per depth: base_train -> base_eval -> base_inference -> chat_train -> chat_eval
#   -> curve.log (the aggregated cost-performance curve, the experiment's product)
# Every stage is idempotent: re-running the same command skips completed work,
# so a crashed run resumes exactly where it left off.
#
# Usage:
#   bash run.sh <experiment_name>
# Examples:
#   bash run.sh jul4_baseline
#   DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8" bash run.sh fast_d24
#   STAGES="base_train base_eval base_inference" bash run.sh pretrain_only   # skip the chat stages
# The run takes hours, so consider a screen session:
#   screen -L -Logfile run.log -S run bash run.sh jul4_baseline

# -----------------------------------------------------------------------------
# Experiment identity

EXPERIMENT_NAME="${1:?usage: bash run.sh <experiment_name>}"
export NANOCHAT_EXPERIMENT="$EXPERIMENT_NAME"

# -----------------------------------------------------------------------------
# Configuration

# 1 thread per process is right when GPUs do the math; CPU presets override this
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# keep python stdout live on the console even though it flows through `tee` pipes
export PYTHONUNBUFFERED=1
# shared cache for immutable artifacts: dataset shards, eval bundle
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
# everything this experiment produces lands here (kept out of the source tree)
EXPERIMENT_DIR="$NANOCHAT_BASE_DIR/experiments/$EXPERIMENT_NAME"

# the dataset this experiment trains on, resolved to $NANOCHAT_BASE_DIR/datasets/<name>.
# the default (climbmix) is canonical and downloaded on demand; any other name is a
# user-provided directory of parquet shards (see harness/dataset.py for the contract).
export NANOCHAT_DATASET="${NANOCHAT_DATASET:-climbmix}"

# number of pretraining data shards to download and pack, ~100MB (~48M tokens) each.
# already-prepared shards are skipped, so this is fast when the cache is warm.
# the default covers the largest rung of the default ladder: d28 consumes ~287
# shards at the default param:data ratio of 12 (estimate includes best-fit cropping
# waste; see dev/estimate_shards_needed.py). bump it if you grow the ladder OR raise
# the ratio: demand scales linearly with it, e.g. ratio 20 at d28 needs ~480 shards.
NUM_SHARDS="${NUM_SHARDS:-300}"

# the depth ladder: one model is trained per depth, tracing out the cost-perf curve.
# the default spans exactly 1e18 -> 1e20 FLOPs (~13 hours total on 8xH100), denser at
# the cheap end: 12 14 16 alone form a <1h mini-ladder for quick scaling checks.
# drop the 28 for an overnight-sized run (~6 hours)
DEPTHS="${DEPTHS:-12 14 16 20 24 28}"
# which stages to run at each depth (e.g. a pretraining researcher: STAGES="base_train base_eval base_inference")
STAGES="${STAGES:-base_train base_eval base_inference chat_train chat_eval}"
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
# sentinel files: live-edit a running (or resumed) experiment by touching files in
# its directory. The ladder is a grid of (depth, stage) units; a unit runs iff it is
# planned (DEPTHS/STAGES), not already done (summary record), and not skipped:
#   touch $EXPERIMENT_DIR/skip_d24        # skip a depth (all its stages)
#   touch $EXPERIMENT_DIR/skip_chat_train      # skip a stage (at all depths)
#   touch $EXPERIMENT_DIR/skip_d24_chat_train  # skip one unit
#   touch $EXPERIMENT_DIR/stop            # start no new units; still aggregate
# A running script notices at the next unit boundary. Aggregation (curve.log)
# always runs, so stopping/skipping still yields a complete curve of what exists.
# rm the file to undo; the next (re)run fills the gap (units are idempotent).
stopped() { [ -f "$EXPERIMENT_DIR/stop" ]; }
skipped() { [ -f "$EXPERIMENT_DIR/skip_d$1" ] || [ -f "$EXPERIMENT_DIR/skip_$2" ] || [ -f "$EXPERIMENT_DIR/skip_d$1_$2" ]; }
# gpus to train on
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# explicit number of training steps, for debugging (-1 = compute optimal horizon)
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
# extra flags passed verbatim to base_train, e.g. a single d24 on fewer tokens with fp8 matmuls:
#   DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8" bash run.sh fast_d24
BASE_TRAIN_FLAGS="${BASE_TRAIN_FLAGS:-}"
# the uv dependency extra to sync ("gpu" or "cpu")
UV_EXTRA="${UV_EXTRA:-gpu}"
# wandb run name prefix ("dummy" disables wandb logging)
WANDB_RUN="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# Environment: uv, venv, dependencies (idempotent, fast when already set up)
# SKIP_SETUP=1 skips the sync, e.g. on a dev machine with extra packages installed

if [ -z "$SKIP_SETUP" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra "$UV_EXTRA"
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Initialize the experiment: creates experiments/<name>/ and records identity
# (git commit, dataset, ...) in meta.json. Resumes if it already exists.

python -m harness.experiment init

# -----------------------------------------------------------------------------
# Data: download the raw shards, train the tokenizer, pack the token shards, and (only
# if a chat stage is planned) the SFT rows; all once per dataset in the shared store
# (idempotent per artifact, so fast when warm). The tokenizer lives with the dataset.
# See scripts/base_prepare.py, scripts/chat_prepare.py and nanochat/dataloader.py.
# Each stage's stdout+stderr is tee'd to a .log; the lines matching the record grammar
# (see harness/experiment.py) are the machine-readable results, the rest is prose.

DATASET_DIR="$NANOCHAT_BASE_DIR/datasets/$NANOCHAT_DATASET"
mkdir -p "$DATASET_DIR"
python -m scripts.base_prepare -n "$NUM_SHARDS" 2>&1 | tee -a "$DATASET_DIR/base_prepare.log"
if has_stage chat_train; then
    python -m scripts.chat_prepare 2>&1 | tee -a "$DATASET_DIR/chat_prepare.log"
fi

# -----------------------------------------------------------------------------
# Pretraining: train one base model per depth in the ladder.
# A depth is complete when its log contains a `summary` record; completed depths
# are skipped on re-entry, so a crashed ladder resumes where it left off.
# Checkpoints land in experiments/<name>/d<depth>/base/.

if has_stage base_train; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/base_train.log"
    stopped && { echo "stop file present: starting no new work"; break; }
    skipped "$depth" base_train && { echo "d${depth}: base_train skipped (skip file present)"; continue; }
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: base model already trained, skipping"
        continue
    fi
    mkdir -p "$MODEL_DIR"
    RUN_NAME=$([ "$WANDB_RUN" = "dummy" ] && echo "dummy" || echo "${WANDB_RUN}_d${depth}")
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$depth" \
        --num-iterations="$NUM_ITERATIONS" \
        --run="$RUN_NAME" \
        $BASE_TRAIN_FLAGS \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Base evals: the CORE metric of each base model (the number the curve is judged on) and a few samples

if has_stage base_eval; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/base_eval.log"
    stopped && { echo "stop file present: starting no new work"; break; }
    skipped "$depth" base_eval && { echo "d${depth}: base_eval skipped (skip file present)"; continue; }
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: base_eval already done, skipping"
        continue
    fi
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
        -g "d${depth}" \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Inference bench: latency/throughput/VRAM of each base model (single GPU)

if has_stage base_inference; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/base_inference.log"
    stopped && { echo "stop file present: starting no new work"; break; }
    skipped "$depth" base_inference && { echo "d${depth}: base_inference skipped (skip file present)"; continue; }
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: inference bench already done, skipping"
        continue
    fi
    python -m scripts.base_inference -i base -g "d${depth}" 2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# SFT: finetune each base model into a chat model (conversation tokens, tool use,
# multiple choice). Hyperparameters are inherited from the pretrained checkpoint.
# Checkpoints land in experiments/<name>/d<depth>/chat/.

if has_stage chat_train; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/chat_train.log"
    stopped && { echo "stop file present: starting no new work"; break; }
    skipped "$depth" chat_train && { echo "d${depth}: chat_train skipped (skip file present)"; continue; }
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: chat_train already done, skipping"
        continue
    fi
    RUN_NAME=$([ "$WANDB_RUN" = "dummy" ] && echo "dummy" || echo "${WANDB_RUN}_d${depth}_chat")
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_train -- \
        --model-tag="d${depth}" \
        --num-iterations="$NUM_ITERATIONS" \
        --run="$RUN_NAME" \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Chat evals: the official full evaluation of each chat model (ChatCORE)

if has_stage chat_eval; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/chat_eval.log"
    stopped && { echo "stop file present: starting no new work"; break; }
    skipped "$depth" chat_eval && { echo "d${depth}: chat_eval skipped (skip file present)"; continue; }
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: chat eval already done, skipping"
        continue
    fi
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
        -i chat -g "d${depth}" \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Aggregate: join all stage records into the cost-performance curve.
# This is the product of the experiment: experiments/<name>/curve.log

python -m harness.experiment curve
