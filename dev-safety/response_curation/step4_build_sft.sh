#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export SAFETY_CURATION_DIR="${SAFETY_CURATION_DIR:-${NANOCHAT_BASE_DIR}/safety_response_curation}"

mkdir -p "$SAFETY_CURATION_DIR"

INPUT_FILE="${INPUT_FILE:-${SAFETY_CURATION_DIR}/step3_verifications.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-step4_sft.jsonl}"
META_NAME="${META_NAME:-step4_selected_metadata.jsonl}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SEED="${SEED:-13}"

python3 "$ROOT_DIR/dev-safety/response_curation/step4_build_sft.py" \
  --input "$INPUT_FILE" \
  --output-dir "$SAFETY_CURATION_DIR" \
  --output-name "$OUTPUT_NAME" \
  --meta-name "$META_NAME" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED"
