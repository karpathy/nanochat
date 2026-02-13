#!/bin/bash

# Scaling Laws Sweep for GPT-Gamma + MuonH (Hyperball)
# Runs IsoFLOP analysis: for each compute budget, sweep model depths to find optimal size.
# Results saved to CSV for analysis with dev/scaling_analysis.ipynb
#
# Usage:
#   bash runs/scaling_laws_muonh.sh
#   LABEL=feb06 bash runs/scaling_laws_muonh.sh
#   FP8=0 bash runs/scaling_laws_muonh.sh

set -e

LABEL="${LABEL:-muonh_$(date +%b%d | tr '[:upper:]' '[:lower:]')}"

FLOPS_BUDGETS=(
    1e18
    2.15e18
    4.64e18
    1e19
)
DEPTHS=(8 10 12 14 16 18 20)

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    NPROC_PER_NODE=1
fi

# Fixed batch size (auto batch size requires target-param-data-ratio, not compatible with target-flops)
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval

# Optimizer (MuonH defaults)
MATRIX_OPTIMIZER="${MATRIX_OPTIMIZER:-hyperball}"
MATRIX_LR="${MATRIX_LR:-0.02}"
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
SCALAR_LR="${SCALAR_LR:-0.5}"
NORM_LR="${NORM_LR:-0.2}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.3}"
MATRIX_WARMDOWN_RATIO="${MATRIX_WARMDOWN_RATIO:-1.0}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"

# FP8 (default enabled)
FP8="${FP8:-1}"
FP8_ARGS=""
if [ "${FP8}" -eq 1 ]; then
    FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
    FP8_ARGS="--fp8 --fp8-recipe=${FP8_RECIPE}"
fi

# Wandb
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-scaling}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"

# Paths and cache
export OMP_NUM_THREADS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PROJECT_ROOT/cache}"
export TORCHINDUCTOR_CACHE_DIR="$NANOCHAT_BASE_DIR/torch_inductor"
export TRITON_CACHE_DIR="$NANOCHAT_BASE_DIR/triton"
export TMPDIR="$NANOCHAT_BASE_DIR/tmp"
mkdir -p "$NANOCHAT_BASE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

cd "$PROJECT_ROOT"

# Python venv
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_norm_and_proj_scalars,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local flops=$1
    local depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Print summary
# =============================================================================

log "=============================================="
log "Scaling Laws Sweep (GPT-Gamma + MuonH)"
log "=============================================="
log "Label:             $LABEL"
log "FLOPs budgets:     ${FLOPS_BUDGETS[*]}"
log "Depths:            ${DEPTHS[*]}"
log "Num GPUs:          $NPROC_PER_NODE"
log "Total batch size:  $TOTAL_BATCH_SIZE"
log "Matrix optimizer:  $MATRIX_OPTIMIZER"
log "Matrix LR:         $MATRIX_LR"
log "Norm LR:           $NORM_LR"
log "Warmdown ratio:    adam=$WARMDOWN_RATIO, matrix=$MATRIX_WARMDOWN_RATIO"
if [ "${FP8}" -eq 1 ]; then
    log "FP8:               enabled ($FP8_RECIPE)"
fi
log "Results dir:       $RESULTS_DIR"
log "=============================================="

# =============================================================================
# Main Loop
# =============================================================================

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "Compute budget: $flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        if run_exists "$flops" "$d"; then
            log "Skipping d=$d at $flops FLOPs (already in results)"
            continue
        fi

        log "Training d=$d at $flops FLOPs..."

        # Unique tag for this run
        TAG="scaling_${LABEL}_${flops}_d${d}"

        # Record start time
        START_TIME=$(date +%s)

        # Train the model with fixed flops budget
        TRAIN_ARGS=(
            --depth=$d
            --target-flops=$flops
            --target-param-data-ratio=-1
            --total-batch-size=$TOTAL_BATCH_SIZE
            --device-batch-size=$DEVICE_BATCH_SIZE
            --run="${WANDB_RUN}_${TAG}"
            --model-tag="${TAG}"
            --window-pattern=$WINDOW_PATTERN
            --matrix-optimizer=$MATRIX_OPTIMIZER
            --matrix-lr=$MATRIX_LR
            --embedding-lr=$EMBEDDING_LR
            --unembedding-lr=$UNEMBEDDING_LR
            --scalar-lr=$SCALAR_LR
            --norm-lr=$NORM_LR
            --warmdown-ratio=$WARMDOWN_RATIO
            --matrix-warmdown-ratio=$MATRIX_WARMDOWN_RATIO
            --eval-tokens=$EVAL_TOKENS
            --core-metric-every=999999
            --core-metric-max-per-task=-1
            --sample-every=-1
            --save-every=-1
        )

        if [ "$NPROC_PER_NODE" -gt 1 ]; then
            torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
                "${TRAIN_ARGS[@]}" $FP8_ARGS \
                2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"
        else
            python -m scripts.base_train \
                "${TRAIN_ARGS[@]}" $FP8_ARGS \
                2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"
        fi

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # Extract detailed parameter counts (handle whitespace-padded format)
        PARAMS_WTE=$(grep "wte" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_VE=$(grep "value_embeds" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_LM=$(grep "lm_head" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TRANSFORMER=$(grep "transformer_matrices" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_NORM=$(grep "norm_and_proj_scalars" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_SCALARS=$(grep -w "scalars" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TOTAL=$(grep -w "total" "$LOG_FILE" | grep ":" | tail -1 | grep -oP '[\d,]+' | tr -d ',')

        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        TOKENS_TRAINED=$((NUM_ITERS * TOTAL_BATCH_SIZE))
        MODEL_DIM=$((d * 64))
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # Extract CORE score from training log (evaluated on final step)
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "WARNING: Could not extract CORE score for d=$d"
            CORE_SCORE="0.0"
        fi

        log "  Params: $PARAMS_TOTAL (transformer: $PARAMS_TRANSFORMER), Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        # Append to CSV
        echo "$flops,$d,$MODEL_DIM,$PARAMS_WTE,$PARAMS_VE,$PARAMS_LM,$PARAMS_TRANSFORMER,$PARAMS_NORM,$PARAMS_SCALARS,$PARAMS_TOTAL,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
