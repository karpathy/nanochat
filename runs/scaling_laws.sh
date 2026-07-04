#!/bin/bash
set -e -o pipefail

# Scaling laws sweep: train a grid of (flops budget x depth) models, all inside one
# experiment. For each flops budget, the best depth traces out the compute-optimal
# frontier. Results aggregate into the experiment's curve.log (one `model` record per
# run; the `params`/`summary` records in each base_train.log carry the details).
#
# Usage:
#   bash runs/scaling_laws.sh <experiment_name>
# Example:
#   bash runs/scaling_laws.sh scaling_jul4
# The grid is env-overridable:
#   FLOPS_BUDGETS="1e18 1e19" DEPTHS="10 14 18" bash runs/scaling_laws.sh scaling_jul4

EXPERIMENT_NAME="${1:?usage: bash runs/scaling_laws.sh <experiment_name>}"
export NANOCHAT_EXPERIMENT="$EXPERIMENT_NAME"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DATASET="${NANOCHAT_DATASET:-climbmix}"
EXPERIMENT_DIR="$NANOCHAT_BASE_DIR/experiments/$EXPERIMENT_NAME"

FLOPS_BUDGETS="${FLOPS_BUDGETS:-1e18 2.15e18 4.64e18 1e19}"
DEPTHS="${DEPTHS:-10 12 14 16 18 20}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_SHARDS="${NUM_SHARDS:-1000}"
EVAL_TOKENS=$((100 * 524288)) # ~100M tokens for the final eval (default is ~10M)
WANDB_RUN="${WANDB_RUN:-dummy}"

source .venv/bin/activate

# experiment identity, dataset, tokenizer (idempotent, shared with runs/run.sh conventions)
python -m nanochat.experiment
python -m nanochat.dataset -n "$NUM_SHARDS"
if [ ! -f "$EXPERIMENT_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768 2>&1 | tee "$EXPERIMENT_DIR/tok_train.log"
fi

# the grid: one model per (flops budget, depth), tagged e.g. flops1e18_d12.
# a cell is complete when its log contains a `summary` record, so re-running
# the script resumes wherever it left off.
for flops in $FLOPS_BUDGETS; do
    for depth in $DEPTHS; do
        TAG="flops${flops}_d${depth}"
        MODEL_DIR="$EXPERIMENT_DIR/$TAG"
        LOG="$MODEL_DIR/base_train.log"
        if grep -q "^summary " "$LOG" 2>/dev/null; then
            echo "${TAG}: already trained, skipping"
            continue
        fi
        mkdir -p "$MODEL_DIR"
        RUN_NAME=$([ "$WANDB_RUN" = "dummy" ] && echo "dummy" || echo "${WANDB_RUN}_${TAG}")
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
            --depth="$depth" \
            --target-flops="$flops" \
            --target-param-data-ratio=-1 \
            --model-tag="$TAG" \
            --eval-tokens="$EVAL_TOKENS" \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --run="$RUN_NAME" \
            2>&1 | tee "$LOG"
    done
done

# aggregate all runs into the experiment's curve.log
python -m scripts.curve
