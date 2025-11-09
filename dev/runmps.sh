#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# Stage selection (allow running only a subset, e.g. --stage=sft or --from=mid)
RUN_BASE=1
RUN_MID=1
RUN_SFT=1
RUN_REPORT=1
STAGE_ONLY=""
FROM_STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage=*)
            STAGE_ONLY="${1#*=}"
            ;;
        --base|--mid|--sft|--report)
            STAGE_ONLY="${1#--}"
            ;;
        --from=*)
            FROM_STAGE="${1#*=}"
            ;;
        --from-base|--from-mid|--from-sft)
            FROM_STAGE="${1#--from-}"
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash dev/runmps.sh [options]

Options:
  --stage=<base|mid|sft|report>  Run only the specified stage.
  --from=<base|mid|sft>          Run from the specified stage through the end.
  --help                         Show this help message.

Environment variables (same as before) control batch sizes, WANDB run names, etc.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ -n "$FROM_STAGE" ]]; then
    RUN_BASE=0
    RUN_MID=0
    RUN_SFT=0
    RUN_REPORT=0
    case "$FROM_STAGE" in
        base)
            RUN_BASE=1
            RUN_MID=1
            RUN_SFT=1
            RUN_REPORT=1
            ;;
        mid)
            RUN_MID=1
            RUN_SFT=1
            ;;
        sft)
            RUN_SFT=1
            ;;
        *)
            echo "Unknown --from stage: $FROM_STAGE" >&2
            exit 1
            ;;
    esac
fi

if [[ -n "$STAGE_ONLY" ]]; then
    RUN_BASE=0
    RUN_MID=0
    RUN_SFT=0
    RUN_REPORT=0
    case "$STAGE_ONLY" in
        base)
            RUN_BASE=1
            ;;
        mid)
            RUN_MID=1
            ;;
        sft)
            RUN_SFT=1
            ;;
        report)
            RUN_REPORT=1
            ;;
        *)
            echo "Unknown --stage value: $STAGE_ONLY" >&2
            exit 1
            ;;
    esac
fi

if [[ -n "$STAGE_ONLY" || -n "$FROM_STAGE" ]]; then
    # avoid regenerating reports when running a subset unless specifically requested
    if [[ "$STAGE_ONLY" != "report" && "$FROM_STAGE" != "base" ]]; then
        RUN_REPORT=0
    fi
fi

# all the setup stuff
export OMP_NUM_THREADS=1
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
if [ "$WANDB_RUN" != "dummy" ] && [ -z "$WANDB_MODE" ]; then
    export WANDB_MODE=online
fi

# Batch/sequence configuration
BASE_DEPTH=${BASE_DEPTH:-4}
SEQ_LEN=${SEQ_LEN:-1024}
DEVICE_BATCH=${DEVICE_BATCH:-16}
TOTAL_BATCH=${TOTAL_BATCH:-$((DEVICE_BATCH * SEQ_LEN))} # tokens per optimizer step
KV_HEAD_MULT=${KV_HEAD_MULT:-1}
EVAL_SEQUENCES=10000
EVAL_STEPS=$(((EVAL_SEQUENCES + DEVICE_BATCH - 1) / DEVICE_BATCH))
EVAL_BATCH_MULT=4 # evaluate on 4 full batches
EVAL_TOKENS=$((TOTAL_BATCH * EVAL_BATCH_MULT))
MID_NUM_STEPS=6144
SFT_NUM_STEPS=${SFT_NUM_STEPS:-3072}
CHECKPOINT_EVERY_SEQ=${CHECKPOINT_EVERY_SEQ:-10000}
RUN_STAGE_EVALS=${RUN_STAGE_EVALS:-0}
WANDB_PROJECT=${WANDB_PROJECT:-nanochat}
TARGET_PARAM_DATA_RATIO=${TARGET_PARAM_DATA_RATIO:-20}
SFT_DEVICE_BATCH=${SFT_DEVICE_BATCH:-$DEVICE_BATCH}
SFT_TARGET_EXAMPLES=${SFT_TARGET_EXAMPLES:-$DEVICE_BATCH}
SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-0}
SFT_EVAL_STEPS=${SFT_EVAL_STEPS:-0}
SFT_EVAL_METRICS_EVERY=${SFT_EVAL_METRICS_EVERY:-0}
SFT_EVAL_METRICS_MAX=${SFT_EVAL_METRICS_MAX:-0}

STATE_FILE=".runmps_wandb_ids"
printf '' > "$STATE_FILE"
echo "WANDB_PROJECT=$WANDB_PROJECT" >> "$STATE_FILE"

generate_wandb_id() {
    python - <<'PY'
import uuid
print(uuid.uuid4().hex[:8])
PY
}

BASE_SEQS_PER_STEP=$((TOTAL_BATCH / SEQ_LEN))
if [ $BASE_SEQS_PER_STEP -le 0 ]; then BASE_SEQS_PER_STEP=1; fi
if [ "$CHECKPOINT_EVERY_SEQ" -le 0 ]; then
    BASE_CHECKPOINT_STEPS=0
else
    BASE_CHECKPOINT_STEPS=$(((CHECKPOINT_EVERY_SEQ + BASE_SEQS_PER_STEP - 1) / BASE_SEQS_PER_STEP))
fi

MID_SEQS_PER_STEP=$((TOTAL_BATCH / SEQ_LEN))
if [ $MID_SEQS_PER_STEP -le 0 ]; then MID_SEQS_PER_STEP=1; fi
if [ "$CHECKPOINT_EVERY_SEQ" -le 0 ]; then
    MID_CHECKPOINT_STEPS=0
else
    MID_CHECKPOINT_STEPS=$(((CHECKPOINT_EVERY_SEQ + MID_SEQS_PER_STEP - 1) / MID_SEQS_PER_STEP))
fi

SFT_SEQS_PER_STEP=$SFT_DEVICE_BATCH
if [ $SFT_SEQS_PER_STEP -le 0 ]; then SFT_SEQS_PER_STEP=1; fi
if [ "$CHECKPOINT_EVERY_SEQ" -le 0 ]; then
    SFT_CHECKPOINT_STEPS=0
else
    SFT_CHECKPOINT_STEPS=$(((CHECKPOINT_EVERY_SEQ + SFT_SEQS_PER_STEP - 1) / SFT_SEQS_PER_STEP))
fi

# Auto-populate WANDB_API_KEY from ~/.netrc when talking to a local W&B server.
# Mirrors the helper used in TinyRecursiveModels/pretrain_text.py so we can log
# to a self-hosted instance without manual export each time.
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
    # Allow custom WANDB_BASE_URL; default to localhost if user sets WANDB
    WANDB_BASE_URL_DEFAULT=${WANDB_BASE_URL:-http://localhost:8080}
    if printf '%s' "$WANDB_BASE_URL_DEFAULT" | grep -q "localhost"; then
        HOST=$(printf '%s\n' "$WANDB_BASE_URL_DEFAULT" | sed -E 's#https?://##' | cut -d/ -f1)
        API_KEY=$(python - "$HOST" <<'PY'
import sys
from netrc import netrc

host = sys.argv[1]
auth = netrc().authenticators(host)
if auth and auth[2]:
    print(auth[2], end="")
PY
)
        if [ -n "$API_KEY" ]; then
            export WANDB_BASE_URL="$WANDB_BASE_URL_DEFAULT"
            export WANDB_API_KEY="$API_KEY"
            echo "[runmps] Loaded WANDB_API_KEY for $WANDB_BASE_URL from ~/.netrc"
        fi
    fi
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

if (( RUN_BASE )); then
    # wipe the report
    python -m nanochat.report reset

# train tokenizer on ~2B characters (download full shard set for extended training)
python -m nanochat.dataset -n 240
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

    # train a very small 4 layer model on the CPU
    # each optimization step processes a single sequence of 1024 tokens
    # we only run 50 steps of optimization (bump this to get better results)
    if [ "$WANDB_RUN" != "dummy" ]; then
        BASE_WANDB_ID=${BASE_WANDB_ID:-$(generate_wandb_id)}
        echo "BASE_WANDB_ID=$BASE_WANDB_ID" >> "$STATE_FILE"
        echo "BASE_WANDB_NAME=$WANDB_RUN" >> "$STATE_FILE"
        export WANDB_RUN_ID=$BASE_WANDB_ID
        export WANDB_EVAL_RUN=$WANDB_RUN
        export WANDB_PROJECT
    fi

    python -m scripts.base_train \
        --depth=$BASE_DEPTH \
        --max_seq_len=$SEQ_LEN \
        --device_batch_size=$DEVICE_BATCH \
        --total_batch_size=$TOTAL_BATCH \
        --kv_head_mult=$KV_HEAD_MULT \
        --target_param_data_ratio=$TARGET_PARAM_DATA_RATIO \
        --run="$WANDB_RUN" \
        --eval_every=$EVAL_STEPS \
        --eval_tokens=$EVAL_TOKENS \
        --core_metric_every=-1 \
        --sample_every=-1 \
        --checkpoint_every_steps=$BASE_CHECKPOINT_STEPS

    if [ "$WANDB_RUN" != "dummy" ]; then
        unset WANDB_RUN_ID
        unset WANDB_EVAL_RUN
    fi

    if [ "$RUN_STAGE_EVALS" = "1" ]; then
        python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH --split_tokens=$EVAL_TOKENS
        python -m scripts.base_eval --max-per-task=16
    fi
fi

if (( RUN_MID )); then
    # midtraining
    if [ "$WANDB_RUN" != "dummy" ]; then
        MID_WANDB_ID=${MID_WANDB_ID:-$(generate_wandb_id)}
        echo "MID_WANDB_ID=$MID_WANDB_ID" >> "$STATE_FILE"
        echo "MID_WANDB_NAME=${WANDB_RUN}-mid" >> "$STATE_FILE"
        export WANDB_RUN_ID=$MID_WANDB_ID
        export WANDB_EVAL_RUN="${WANDB_RUN}-mid"
        export WANDB_PROJECT
    fi

    python -m scripts.mid_train \
        --max_seq_len=$SEQ_LEN \
        --device_batch_size=$DEVICE_BATCH \
        --total_batch_size=$TOTAL_BATCH \
        --run="${WANDB_RUN}-mid" \
        --eval_every=$EVAL_STEPS \
        --eval_tokens=$EVAL_TOKENS \
        --checkpoint_every_steps=$MID_CHECKPOINT_STEPS \
        --num_iterations=$MID_NUM_STEPS
    if [ "$WANDB_RUN" != "dummy" ]; then
        unset WANDB_RUN_ID
        unset WANDB_EVAL_RUN
    fi
    if [ "$RUN_STAGE_EVALS" = "1" ]; then
        # eval results will be terrible, this is just to execute the code paths.
        # note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
        python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20
    fi
fi

if (( RUN_SFT )); then
    # SFT
    if [ "$WANDB_RUN" != "dummy" ]; then
        SFT_WANDB_ID=${SFT_WANDB_ID:-$(generate_wandb_id)}
        echo "SFT_WANDB_ID=$SFT_WANDB_ID" >> "$STATE_FILE"
        echo "SFT_WANDB_NAME=${WANDB_RUN}-sft" >> "$STATE_FILE"
        export WANDB_RUN_ID=$SFT_WANDB_ID
        export WANDB_EVAL_RUN="${WANDB_RUN}-sft"
        export WANDB_PROJECT
    fi

    python -m scripts.chat_sft \
        --device_batch_size=$SFT_DEVICE_BATCH \
        --target_examples_per_step=$SFT_TARGET_EXAMPLES \
        --run="${WANDB_RUN}-sft" \
        --num_iterations=$SFT_NUM_STEPS \
        --eval_every=$SFT_EVAL_EVERY \
        --eval_steps=$SFT_EVAL_STEPS \
        --eval_metrics_every=$SFT_EVAL_METRICS_EVERY \
        --eval_metrics_max_problems=$SFT_EVAL_METRICS_MAX \
        --checkpoint_every_steps=$SFT_CHECKPOINT_STEPS

    if [ "$WANDB_RUN" != "dummy" ]; then
        unset WANDB_RUN_ID
        unset WANDB_EVAL_RUN
    fi
fi

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

if (( RUN_REPORT )); then
    python -m nanochat.report generate
fi

if [ "$RUN_STAGE_EVALS" != "1" ] && (( RUN_BASE || RUN_MID || RUN_SFT )); then
    echo "[runmps] Inline evals disabled. Run 'bash dev/runmps_evals.sh' to compute metrics from saved checkpoints."
fi
