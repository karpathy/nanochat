#!/bin/bash
# -e: stop on any stage failure. pipefail: a failure inside `python ... | tee` must
# surface as the pipeline's exit code (otherwise tee's success would mask it).
set -e -o pipefail

# The nanochat master script: runs one experiment, end to end.
#
# An experiment = (name, git commit, depth ladder, dataset). Everything derives from
# --depth, so there is no config file: the code is the config.
# See experiment_refactor.md for the full design. The pipeline:
#   setup -> experiment identity (meta.json) -> dataset -> tokenizer
#   -> per depth: base_train -> infer_bench -> sft -> chat_eval
#   -> curve.log (the aggregated cost-performance curve, the experiment's product)
# Every stage is idempotent: re-running the same command skips completed work,
# so a crashed run resumes exactly where it left off.
#
# Usage:
#   bash runs/run.sh <experiment_name>
# Examples:
#   bash runs/run.sh jul4_baseline
#   DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8" bash runs/run.sh speedrun
#   STAGES="base infer" bash runs/run.sh pretrain_only   # skip the post-training stages
# The run takes hours, so consider a screen session:
#   screen -L -Logfile run.log -S run bash runs/run.sh jul4_baseline

# -----------------------------------------------------------------------------
# Experiment identity

EXPERIMENT_NAME="${1:?usage: bash runs/run.sh <experiment_name>}"
export NANOCHAT_EXPERIMENT="$EXPERIMENT_NAME"

# -----------------------------------------------------------------------------
# Configuration

export OMP_NUM_THREADS=1
# keep python stdout live on the console even though it flows through `tee` pipes
export PYTHONUNBUFFERED=1
# shared cache for immutable artifacts: dataset shards, eval bundle
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
# everything this experiment produces lands here (kept out of the source tree)
EXPERIMENT_DIR="$NANOCHAT_BASE_DIR/experiments/$EXPERIMENT_NAME"

# the dataset this experiment trains on, resolved to $NANOCHAT_BASE_DIR/datasets/<name>.
# the default (climbmix) is canonical and downloaded on demand; any other name is a
# user-provided directory of parquet shards (see nanochat/dataset.py for the contract).
export NANOCHAT_DATASET="${NANOCHAT_DATASET:-climbmix}"

# number of pretraining data shards to download, ~100MB of compressed text each.
# already-downloaded shards are skipped, so this is fast when the cache is warm.
# TODO: derive from the ladder (the largest depth determines how many are consumed)
NUM_SHARDS="${NUM_SHARDS:-1000}"

# the depth ladder: one model is trained per depth, tracing out the cost-perf curve
DEPTHS="${DEPTHS:-12 16 20 24}"
# which stages to run at each depth (e.g. a pretraining researcher: STAGES="base infer")
STAGES="${STAGES:-base infer sft chat}"
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
# gpus to train on
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# explicit number of training steps, for debugging (-1 = compute optimal horizon)
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
# extra flags passed verbatim to base_train, e.g. the leaderboard speedrun is:
#   DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8" bash runs/run.sh speedrun
BASE_TRAIN_FLAGS="${BASE_TRAIN_FLAGS:-}"
# wandb run name prefix ("dummy" disables wandb logging)
WANDB_RUN="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# Environment: uv, venv, dependencies (idempotent, fast when already set up)
# SKIP_SETUP=1 skips the sync, e.g. on a dev machine with extra packages installed

if [ -z "$SKIP_SETUP" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Initialize the experiment: creates experiments/<name>/ and records identity
# (git commit, dataset, ...) in meta.json. Resumes if it already exists.

python -m nanochat.experiment

# -----------------------------------------------------------------------------
# Pretraining data: materialize/verify the experiment's dataset (shared store,
# reused across experiments; quiet and fast when the cache is warm)

python -m nanochat.dataset -n "$NUM_SHARDS"

# -----------------------------------------------------------------------------
# Tokenizer: trained on the experiment's dataset, lives in the experiment dir.
# Skipped if this experiment already trained it (idempotent re-entry).
# Each stage's stdout+stderr is tee'd to a .log in the experiment dir; the lines
# matching the record grammar (see nanochat/logfmt.py) are the machine-readable
# results, everything else is human-facing prose.

if [ ! -f "$EXPERIMENT_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 2>&1 | tee "$EXPERIMENT_DIR/tok_train.log"
    python -m scripts.tok_eval 2>&1 | tee "$EXPERIMENT_DIR/tok_eval.log"
fi

# -----------------------------------------------------------------------------
# Pretraining: train one base model per depth in the ladder.
# A depth is complete when its log contains a `summary` record; completed depths
# are skipped on re-entry, so a crashed ladder resumes where it left off.
# Checkpoints land in experiments/<name>/d<depth>/base/.

if has_stage base; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/base_train.log"
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: base model already trained, skipping"
        continue
    fi
    mkdir -p "$MODEL_DIR"
    RUN_NAME=$([ "$WANDB_RUN" = "dummy" ] && echo "dummy" || echo "${WANDB_RUN}_d${depth}")
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$depth" \
        --num-iterations="$NUM_ITERATIONS" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --run="$RUN_NAME" \
        $BASE_TRAIN_FLAGS \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Inference bench: latency/throughput/VRAM of each base model (single GPU)

if has_stage infer; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/infer_bench.log"
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: inference bench already done, skipping"
        continue
    fi
    python -m scripts.infer_bench -i base -g "d${depth}" 2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# SFT: finetune each base model into a chat model (conversation tokens, tool use,
# multiple choice). Hyperparameters are inherited from the pretrained checkpoint.
# Checkpoints land in experiments/<name>/d<depth>/sft/.

if has_stage sft; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/sft.log"
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: sft already done, skipping"
        continue
    fi
    RUN_NAME=$([ "$WANDB_RUN" = "dummy" ] && echo "dummy" || echo "${WANDB_RUN}_d${depth}_sft")
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
        --model-tag="d${depth}" \
        --num-iterations="$NUM_ITERATIONS" \
        --run="$RUN_NAME" \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Chat evals: the official full evaluation of each chat model (ChatCORE)

if has_stage chat; then
for depth in $DEPTHS; do
    MODEL_DIR="$EXPERIMENT_DIR/d${depth}"
    LOG="$MODEL_DIR/chat_eval.log"
    if grep -q "^summary " "$LOG" 2>/dev/null; then
        echo "d${depth}: chat eval already done, skipping"
        continue
    fi
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
        -i sft -g "d${depth}" \
        2>&1 | tee "$LOG"
done
fi

# -----------------------------------------------------------------------------
# Aggregate: join all stage records into the cost-performance curve.
# This is the product of the experiment: experiments/<name>/curve.log

python -m scripts.curve
