#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export SAFETY_CURATION_DIR="${SAFETY_CURATION_DIR:-${NANOCHAT_BASE_DIR}/safety_response_curation}"
export BASE_URL="${BASE_URL:-https://integrate.api.nvidia.com/v1}"
export API_KEY=""

mkdir -p "$SAFETY_CURATION_DIR"

INPUT_FILE="${INPUT_FILE:-${SAFETY_CURATION_DIR}/step1_coconot_safety_sample.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-step2_generations.jsonl}"
N_RESPONSES="${N_RESPONSES:-2}"
WORKERS="${WORKERS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

python3 "$ROOT_DIR/dev-safety/response_curation/step2_generate_responses.py" \
  --input "$INPUT_FILE" \
  --output-dir "$SAFETY_CURATION_DIR" \
  --output-name "$OUTPUT_NAME" \
  --n-responses "$N_RESPONSES" \
  --workers "$WORKERS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --base-url "$BASE_URL" \
  