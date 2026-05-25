#!/bin/bash
set -euo pipefail

# End-to-end d36 (~3.8B parameter) nanochat run on ClimbMix.
# Designed for a WATGPU H200 batch allocation. Override settings with env vars.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

MODEL_DEPTH="${MODEL_DEPTH:-36}"
MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$DEVICE_BATCH_SIZE}"
TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
DATASET_SHARDS="${DATASET_SHARDS:-1000}"
DATASET_WORKERS="${DATASET_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
USE_FP8="${USE_FP8:-1}"
PREFETCH_TASK_DATA="${PREFETCH_TASK_DATA:-1}"
WANDB_RUN="${WANDB_RUN:-dummy}"
STOP_AFTER="${STOP_AFTER:-full}"

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

if [ "${SKIP_SETUP:-0}" != "1" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

python -m nanochat.report reset

log "Downloading tokenizer seed shards: $TOKENIZER_SHARDS train shard(s) plus validation"
python -m nanochat.dataset -n "$TOKENIZER_SHARDS" -w "$DATASET_WORKERS"

log "Downloading ClimbMix train shards in background: DATASET_SHARDS=$DATASET_SHARDS"
python -m nanochat.dataset -n "$DATASET_SHARDS" -w "$DATASET_WORKERS" &
DATASET_DOWNLOAD_PID=$!

log "Training tokenizer"
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
python -m scripts.tok_eval

log "Downloading identity conversations"
curl -L --fail -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

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
    --run="$WANDB_RUN"
)
if [ "$USE_FP8" = "1" ]; then
    BASE_TRAIN_ARGS+=(--fp8)
fi
if [ -n "${EXTRA_BASE_TRAIN_ARGS:-}" ]; then
    read -r -a extra_base_train_args <<< "$EXTRA_BASE_TRAIN_ARGS"
    BASE_TRAIN_ARGS+=("${extra_base_train_args[@]}")
fi

log "Pretraining base model"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- "${BASE_TRAIN_ARGS[@]}"
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
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"
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
