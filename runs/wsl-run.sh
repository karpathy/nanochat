#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$REPO_ROOT/output}"
mkdir -p "$NANOCHAT_BASE_DIR" "$REPO_ROOT/logs"
exec > >(tee -a "$REPO_ROOT/logs/wsl-run.log") 2>&1
echo "Starting rustAgent run at $(date -Is)"
echo "Live progress: tail -F logs/wsl-run.log"

cargo build --release --features "${RUST_AGENT_FEATURES:-cuda}"
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
WANDB_RUN="${WANDB_RUN:-dummy}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d8}"
BASE_SEQ_LEN="${BASE_SEQ_LEN:-8192}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-1}"
BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-1048576}"
SFT_SEQ_LEN="${SFT_SEQ_LEN:-$BASE_SEQ_LEN}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_STANDARD_TAG="${BASE_MODEL_TAG}-smoltalk"
SFT_TOOLS_TAG="${BASE_MODEL_TAG}-tools"
SFT_POST_TAG="${BASE_MODEL_TAG}-post-training"
SFT_TRACES_TAG="${BASE_MODEL_TAG}-traces"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rust-agent-wsl-smoke.XXXXXX")"
  trap 'rm -rf "$SMOKE_DIR"' EXIT
  for i in $(seq 1 128); do
    printf 'Rust agent smoke corpus line %s trains language models.\n' "$i"
  done > "$SMOKE_DIR/train.txt"

  "$RUST_AGENT" train-tokenizer --dataset "$SMOKE_DIR/train.txt" \
    --vocab-size 300 --max-chars 100000 --doc-cap 200 \
    --output "$SMOKE_DIR/tokenizer.json"
  "$RUST_AGENT" train --dataset "$SMOKE_DIR/train.txt" \
    --tokenizer "$SMOKE_DIR/tokenizer.json" --depth 2 --aspect-ratio 24 \
    --head-dim 24 --max-seq-len 8 --steps 1 --batch-size 1 \
    --output "$SMOKE_DIR/checkpoint" --eval-every 0 \
    --core-metric-every -1 --sample-every -1 --save-every -1 --device cpu
  "$RUST_AGENT" generate --config "$SMOKE_DIR/checkpoint/meta_000001.json" \
    --checkpoint "$SMOKE_DIR/checkpoint/model_000001.pt" \
    --tokenizer "$SMOKE_DIR/tokenizer.json" --prompt "" \
    --max-tokens 4 --temperature 0 --device cpu
  echo "rustAgent WSL smoke run passed"
  exit 0
fi

"$RUST_AGENT" download-dataset --dataset crowdmind_2048
"$RUST_AGENT" download-dataset --dataset climbmix --num-files "${CLIMBMIX_FILES:-170}"
TOKENIZER_PATH="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.json"
if [[ "${REUSE_TOKENIZER:-1}" == "1" && -s "$TOKENIZER_PATH" ]]; then
  echo "reusing tokenizer: $TOKENIZER_PATH"
else
  "$RUST_AGENT" train-tokenizer --dataset climbmix
fi

BASE_TRAIN_ARGS=(--depth "${BASE_DEPTH:-8}" --max-seq-len "$BASE_SEQ_LEN"
  --batch-size "$BASE_BATCH_SIZE" --save-every 100 --dataset climbmix
  --model-tag "$BASE_MODEL_TAG" --run "$WANDB_RUN" --device auto)
if [[ "${BASE_FP8:-1}" == "1" ]]; then
  BASE_TRAIN_ARGS+=(--fp8)
fi
if [[ -n "${BASE_TOTAL_BATCH_SIZE:-}" ]]; then
  BASE_TRAIN_ARGS+=(--total-batch-size "$BASE_TOTAL_BATCH_SIZE")
fi
if [[ -n "${BASE_EVAL_EVERY:-}" ]]; then
  BASE_TRAIN_ARGS+=(--eval-every "$BASE_EVAL_EVERY")
fi
if [[ -n "${BASE_EVAL_TOKENS:-}" ]]; then
  BASE_TRAIN_ARGS+=(--eval-tokens "$BASE_EVAL_TOKENS")
fi
if [[ -n "${BASE_CORE_EVERY:-}" ]]; then
  BASE_TRAIN_ARGS+=(--core-metric-every "$BASE_CORE_EVERY")
fi
if [[ -n "${BASE_SAMPLE_EVERY:-}" ]]; then
  BASE_TRAIN_ARGS+=(--sample-every "$BASE_SAMPLE_EVERY")
fi
if [[ -n "${BASE_STEPS:-}" ]]; then
  BASE_TRAIN_ARGS+=(--steps "$BASE_STEPS")
else
  BASE_TRAIN_ARGS+=(--target-param-data-ratio "${TARGET_PARAM_DATA_RATIO:-8}")
fi
echo "Base training: depth=${BASE_DEPTH:-8} seq_len=$BASE_SEQ_LEN batch_size=$BASE_BATCH_SIZE total_batch_size=$BASE_TOTAL_BATCH_SIZE fp8=${BASE_FP8:-1}"
"$RUST_AGENT" train "${BASE_TRAIN_ARGS[@]}"
"$RUST_AGENT" base-eval --model-tag "$BASE_MODEL_TAG" \
  --device-batch-size "$BASE_BATCH_SIZE" --eval "${BASE_EVAL_TASKS:-core,bpb,sample}" --device auto

SFT_TRAIN_ARGS=(--max-seq-len "$SFT_SEQ_LEN" --batch-size "$SFT_BATCH_SIZE"
  --run "$WANDB_RUN" --device auto)
if [[ -n "${SFT_TOTAL_BATCH_SIZE:-}" ]]; then
  SFT_TRAIN_ARGS+=(--total-batch-size "$SFT_TOTAL_BATCH_SIZE")
fi
if [[ -n "${SFT_EVAL_EVERY:-}" ]]; then
  SFT_TRAIN_ARGS+=(--eval-every "$SFT_EVAL_EVERY")
fi
if [[ -n "${SFT_CHATCORE_EVERY:-}" ]]; then
  SFT_TRAIN_ARGS+=(--chatcore-every "$SFT_CHATCORE_EVERY")
fi
if [[ -n "${SFT_STEPS:-}" ]]; then
  SFT_TRAIN_ARGS+=(--steps "$SFT_STEPS")
fi
"$RUST_AGENT" train --source base --model-tag "$BASE_MODEL_TAG" --output-model-tag "$SFT_STANDARD_TAG" \
  --data-mode standard "${SFT_TRAIN_ARGS[@]}"
"$RUST_AGENT" train --source sft --model-tag "$SFT_STANDARD_TAG" --output-model-tag "$SFT_TOOLS_TAG" \
  --data-mode tools --load-optimizer 0 "${SFT_TRAIN_ARGS[@]}"
"$RUST_AGENT" train --source sft --model-tag "$SFT_TOOLS_TAG" --output-model-tag "$SFT_POST_TAG" \
  --data-mode post-training --load-optimizer 0 "${SFT_TRAIN_ARGS[@]}"
"$RUST_AGENT" train --source sft --model-tag "$SFT_POST_TAG" --output-model-tag "$SFT_TRACES_TAG" \
  --data-mode traces --load-optimizer 0 "${SFT_TRAIN_ARGS[@]}"
EVAL_ARGS=(--device auto)
if [[ -n "${EVAL_MAX_PROBLEMS:-}" ]]; then
  EVAL_ARGS+=(--max-problems "$EVAL_MAX_PROBLEMS")
fi
if [[ -n "${EVAL_MAX_NEW_TOKENS:-}" ]]; then
  EVAL_ARGS+=(--max-new-tokens "$EVAL_MAX_NEW_TOKENS")
fi
"$RUST_AGENT" eval-chat --source sft --model-tag "$SFT_TRACES_TAG" "${EVAL_ARGS[@]}"
