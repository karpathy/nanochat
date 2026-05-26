#!/usr/bin/env bash

# Tiny end-to-end artifact preflight.
#
# This is not meant to train a useful model. It verifies that nanochat can write
# the artifacts you must preserve after an expensive run: tokenizer files,
# checkpoints, optimizer state, metadata, eval output in the report directory.
#
# Usage:
#   export NANOCHAT_BASE_DIR=/persistent/path/nanochat-cache
#   bash runs/preflight_artifacts.sh
#
# The script writes to a timestamped preflight subdirectory under
# NANOCHAT_BASE_DIR, so it does not overwrite real training artifacts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TARGET_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
PREFLIGHT_STAMP="$(date +%Y%m%d_%H%M%S)"
PREFLIGHT_BASE_DIR="${PREFLIGHT_BASE_DIR:-$TARGET_BASE_DIR/preflight_artifacts_$PREFLIGHT_STAMP}"
export NANOCHAT_BASE_DIR="$PREFLIGHT_BASE_DIR"

MODEL_TAG="${MODEL_TAG:-preflight_d2}"
DEVICE_TYPE="${DEVICE_TYPE:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
EVAL_TOKENS="${EVAL_TOKENS:-256}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-5000000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
STEP_PADDED="$(printf "%06d" "$NUM_ITERATIONS")"

device_arg=()
if [[ -n "$DEVICE_TYPE" ]]; then
  device_arg=(--device-type="$DEVICE_TYPE")
fi

echo "nanochat artifact preflight"
echo "repo:               $ROOT_DIR"
echo "target base dir:    $TARGET_BASE_DIR"
echo "preflight base dir: $NANOCHAT_BASE_DIR"
echo "model tag:          $MODEL_TAG"
echo

python - <<'PY'
from nanochat.common import get_base_dir, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
import torch
print(f"nanochat base dir: {get_base_dir()}")
print(f"torch cuda:        {torch.cuda.is_available()}")
print(f"compute dtype:     {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
PY

python -m nanochat.report reset

python -m nanochat.dataset -n 1

python -m scripts.tok_train \
  --max-chars="$TOKENIZER_MAX_CHARS" \
  --vocab-size="$VOCAB_SIZE"

python -m scripts.base_train \
  "${device_arg[@]}" \
  --depth=2 \
  --head-dim=64 \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --total-batch-size="$TOTAL_BATCH_SIZE" \
  --num-iterations="$NUM_ITERATIONS" \
  --eval-every="$NUM_ITERATIONS" \
  --eval-tokens="$EVAL_TOKENS" \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --window-pattern=L \
  --model-tag="$MODEL_TAG" \
  --run=dummy

python -m scripts.base_eval \
  "${device_arg[@]}" \
  --device-batch-size=1 \
  --split-tokens="$MAX_SEQ_LEN" \
  --eval=bpb,sample \
  --model-tag="$MODEL_TAG" \
  --step="$NUM_ITERATIONS"

missing=()
require_file() {
  if [[ ! -f "$1" ]]; then
    missing+=("$1")
  fi
}
require_glob() {
  local pattern="$1"
  if ! compgen -G "$pattern" > /dev/null; then
    missing+=("$pattern")
  fi
}

require_glob "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_*.parquet"
require_file "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
require_file "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"
require_file "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/model_$STEP_PADDED.pt"
require_file "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/meta_$STEP_PADDED.json"
require_file "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/optim_${STEP_PADDED}_rank0.pt"
require_file "$NANOCHAT_BASE_DIR/report/header.md"
require_file "$NANOCHAT_BASE_DIR/report/tokenizer-training.md"
require_file "$NANOCHAT_BASE_DIR/report/base-model-training.md"
require_file "$NANOCHAT_BASE_DIR/report/base-model-evaluation.md"

echo
if (( ${#missing[@]} > 0 )); then
  echo "Preflight failed. Missing expected artifacts:"
  printf '  %s\n' "${missing[@]}"
  exit 1
fi

echo "Preflight passed. Expected artifacts were saved."
echo
echo "Inspect artifact tree:"
echo "  find \"$NANOCHAT_BASE_DIR\" -maxdepth 3 -type f | sort"
echo
echo "Archive this preflight run, exactly like you would archive a paid run:"
echo "  cd \"$NANOCHAT_BASE_DIR\""
echo "  tar -czf ../nanochat-preflight-$PREFLIGHT_STAMP.tar.gz tokenizer base_checkpoints report"
echo
echo "For the real run, preserve the real \$NANOCHAT_BASE_DIR before deleting the machine."
