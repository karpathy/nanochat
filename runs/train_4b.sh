#!/bin/bash
set -euo pipefail

# End-to-end d36 (~3.8B parameter) nanochat run on ClimbMix.
# Designed for a WATGPU H200 batch allocation. Override settings with env vars.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

MODEL_DEPTH="${MODEL_DEPTH:-36}"
MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
SFT_SAVE_EVERY="${SFT_SAVE_EVERY:-25}"
KEEP_LAST_SFT_CHECKPOINTS="${KEEP_LAST_SFT_CHECKPOINTS:-4}"
AUTO_RESUME_SFT="${AUTO_RESUME_SFT:-1}"
TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
DATASET_SHARDS="${DATASET_SHARDS:-1000}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-250}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
USE_FP8="${USE_FP8:-1}"
PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-1}"
SKIP_TOKENIZER_TRAIN="${SKIP_TOKENIZER_TRAIN:-0}"
SKIP_IDENTITY_DOWNLOAD="${SKIP_IDENTITY_DOWNLOAD:-0}"
WANDB_RUN="${WANDB_RUN:-dummy}"
STOP_AFTER="${STOP_AFTER:-full}"
CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:--1}"
AUTO_RESUME="${AUTO_RESUME:-0}"
UV_SYNC_EXTRA="${UV_SYNC_EXTRA:-gpu}"
if [ -z "${NANOCHAT_ENV_DIR:-}" ]; then
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        NANOCHAT_ENV_DIR="$SLURM_TMPDIR/nanochat-venv"
    elif [ -d /dev/shm ]; then
        NANOCHAT_ENV_DIR="/dev/shm/${USER:-nanochat}/nanochat-venv"
    else
        NANOCHAT_ENV_DIR="/tmp/${USER:-nanochat}/nanochat-venv"
    fi
fi
export UV_PROJECT_ENVIRONMENT="$NANOCHAT_ENV_DIR"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [ -z "${JOB_TMP_DIR:-}" ]; then
    if [ -n "${SLURM_TMPDIR:-}" ]; then
        JOB_TMP_DIR="$SLURM_TMPDIR"
    elif [ -d /dev/shm ]; then
        JOB_TMP_DIR="/dev/shm/${USER:-nanochat}/tmp"
    else
        JOB_TMP_DIR="/tmp/${USER:-nanochat}/tmp"
    fi
fi
mkdir -p "$JOB_TMP_DIR"
export TMPDIR="$JOB_TMP_DIR"
export TMP="$JOB_TMP_DIR"
export TEMP="$JOB_TMP_DIR"
if [ "$WANDB_RUN" = "dummy" ]; then
    export WANDB_MODE=disabled
fi

if [ -z "${NPROC_PER_NODE:-}" ]; then
    if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
        NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
        IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
        NPROC_PER_NODE="${#visible_gpus[@]}"
    else
        NPROC_PER_NODE=7
    fi
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting $MODEL_TAG on $NPROC_PER_NODE GPU(s)"
log "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
log "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
log "TMPDIR=$TMPDIR"
if [ -n "$CLIMBMIX_DATA_DIR" ]; then
    export CLIMBMIX_DATA_DIR
    log "CLIMBMIX_DATA_DIR=$CLIMBMIX_DATA_DIR"
fi

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    if ! command -v uv &> /dev/null || ! uv --version &> /dev/null; then
        if [ -z "${UV_INSTALL_DIR:-}" ]; then
            if [ -d /dev/shm ]; then
                UV_INSTALL_DIR="/dev/shm/${USER:-nanochat}/uv-bin"
            else
                UV_INSTALL_DIR="/tmp/${USER:-nanochat}/uv-bin"
            fi
        fi
        export UV_INSTALL_DIR
        mkdir -p "$UV_INSTALL_DIR"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$UV_INSTALL_DIR:$PATH"
    fi
    [ -d "$UV_PROJECT_ENVIRONMENT" ] || uv venv "$UV_PROJECT_ENVIRONMENT"
    uv sync --extra "$UV_SYNC_EXTRA"
fi
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python -m nanochat.report reset

log "Downloading tokenizer seed shards: $TOKENIZER_SHARDS train shard(s) plus validation"
python -m nanochat.dataset -n "$TOKENIZER_SHARDS" -w "$DATASET_WORKERS"

log "Downloading ClimbMix train shards in background: DATASET_SHARDS=$DATASET_SHARDS"
python -m nanochat.dataset -n "$DATASET_SHARDS" -w "$DATASET_WORKERS" &
DATASET_DOWNLOAD_PID=$!

if [ "$SKIP_TOKENIZER_TRAIN" = "1" ]; then
    log "SKIP_TOKENIZER_TRAIN=1, reusing tokenizer in $NANOCHAT_BASE_DIR/tokenizer"
else
    log "Training tokenizer"
    python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS" --vocab-size="$TOKENIZER_VOCAB_SIZE"
    python -m scripts.tok_eval
fi

if [ "$SKIP_IDENTITY_DOWNLOAD" = "1" ] && [ -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    log "SKIP_IDENTITY_DOWNLOAD=1, reusing identity conversations"
else
    log "Downloading identity conversations"
    curl -L --fail -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

if [ "$PREFETCH_TASK_DATA" = "1" ]; then
    log "Prefetching eval bundle and SFT/chat-eval datasets"
    python - <<'PY'
import os

from nanochat.common import get_base_dir, download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee

base_dir = get_base_dir()
if not os.path.exists(os.path.join(base_dir, "eval_bundle")):
    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

tasks_to_touch = [
    ("SmolTalk train", lambda: SmolTalk(split="train")),
    ("SmolTalk test", lambda: SmolTalk(split="test")),
    ("MMLU auxiliary_train", lambda: MMLU(subset="all", split="auxiliary_train")),
    ("MMLU test", lambda: MMLU(subset="all", split="test")),
    ("GSM8K train", lambda: GSM8K(subset="main", split="train")),
    ("GSM8K test", lambda: GSM8K(subset="main", split="test")),
    ("ARC-Easy test", lambda: ARC(subset="ARC-Easy", split="test")),
    ("ARC-Challenge test", lambda: ARC(subset="ARC-Challenge", split="test")),
    ("HumanEval test", lambda: HumanEval()),
    ("SimpleSpelling train", lambda: SimpleSpelling(size=1, split="train")),
    ("SpellingBee test", lambda: SpellingBee(size=1, split="test")),
]

for name, make_task in tasks_to_touch:
    task = make_task()
    if len(task) > 0:
        _ = task[0]
    print(f"Prefetched {name}: {len(task):,} rows")
PY
fi

log "Waiting for ClimbMix download to finish"
wait "$DATASET_DOWNLOAD_PID"

BASE_TRAIN_ARGS=(
    --depth="$MODEL_DEPTH"
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --model-tag="$MODEL_TAG"
    --save-every="$SAVE_EVERY"
    --keep-last-checkpoints="$KEEP_LAST_CHECKPOINTS"
    --run="$WANDB_RUN"
)
if [ "$AUTO_RESUME" = "1" ] && [ "$RESUME_FROM_STEP" = "-1" ]; then
    RESUME_FROM_STEP="$(NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" MODEL_TAG="$MODEL_TAG" python - <<'PY'
import glob
import os

checkpoint_dir = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_checkpoints", os.environ["MODEL_TAG"])
steps = []
for path in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    try:
        steps.append(int(os.path.basename(path).split("_")[-1].split(".")[0]))
    except ValueError:
        pass
print(max(steps) if steps else -1)
PY
)"
    if [ "$RESUME_FROM_STEP" != "-1" ]; then
        log "AUTO_RESUME found checkpoint step $RESUME_FROM_STEP"
    else
        log "AUTO_RESUME found no checkpoint; starting from scratch"
    fi
fi
if [ "$RESUME_FROM_STEP" != "-1" ]; then
    BASE_TRAIN_ARGS+=(--resume-from-step="$RESUME_FROM_STEP")
fi
if [ "$USE_FP8" = "1" ]; then
    BASE_TRAIN_ARGS+=(--fp8)
fi
if [ -n "${EXTRA_BASE_TRAIN_ARGS:-}" ]; then
    read -r -a extra_base_train_args <<< "$EXTRA_BASE_TRAIN_ARGS"
    BASE_TRAIN_ARGS+=("${extra_base_train_args[@]}")
fi

log "Pretraining base model"
BASE_TRAIN_PID=""
forward_base_train_signal() {
    if [ -n "$BASE_TRAIN_PID" ]; then
        log "Forwarding stop signal to base training process $BASE_TRAIN_PID"
        pkill -USR1 -P "$BASE_TRAIN_PID" 2>/dev/null || true
    fi
}
trap forward_base_train_signal USR1 TERM INT
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- "${BASE_TRAIN_ARGS[@]}" &
BASE_TRAIN_PID=$!
set +e
wait "$BASE_TRAIN_PID"
BASE_TRAIN_STATUS=$?
set -e
BASE_TRAIN_PID=""
trap - USR1 TERM INT
if [ "$BASE_TRAIN_STATUS" -ne 0 ]; then
    exit "$BASE_TRAIN_STATUS"
fi
if [ "$STOP_AFTER" = "base_train" ]; then
    log "STOP_AFTER=base_train, stopping after pretraining"
    exit 0
fi

log "Evaluating base model"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$EVAL_DEVICE_BATCH_SIZE"
if [ "$STOP_AFTER" = "base_eval" ]; then
    log "STOP_AFTER=base_eval, stopping after base evaluation"
    exit 0
fi

log "Running SFT"
SFT_ARGS=(
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --save-every="$SFT_SAVE_EVERY" \
    --keep-last-checkpoints="$KEEP_LAST_SFT_CHECKPOINTS" \
    --run="$WANDB_RUN"
)
if [ "$AUTO_RESUME_SFT" = "1" ]; then
    SFT_RESUME_FROM_STEP="$(NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" MODEL_TAG="$MODEL_TAG" python - <<'PY'
import glob
import os

checkpoint_dir = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints", os.environ["MODEL_TAG"])
steps = []
for path in glob.glob(os.path.join(checkpoint_dir, "model_*.pt")):
    try:
        steps.append(int(os.path.basename(path).split("_")[-1].split(".")[0]))
    except ValueError:
        pass
print(max(steps) if steps else -1)
PY
)"
    if [ "$SFT_RESUME_FROM_STEP" != "-1" ]; then
        log "AUTO_RESUME_SFT found checkpoint step $SFT_RESUME_FROM_STEP"
        SFT_ARGS+=(--resume-from-step="$SFT_RESUME_FROM_STEP")
    else
        log "AUTO_RESUME_SFT found no checkpoint; starting SFT from base model"
    fi
fi
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- "${SFT_ARGS[@]}"
if [ "$STOP_AFTER" = "sft" ]; then
    log "STOP_AFTER=sft, stopping after SFT"
    exit 0
fi

log "Evaluating chat model"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft \
    -g "$MODEL_TAG" \
    -b "$SFT_DEVICE_BATCH_SIZE"

python -m nanochat.report generate
log "Done. Report copied to report.md"
