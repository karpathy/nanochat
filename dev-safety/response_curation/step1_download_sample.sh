#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export SAFETY_CURATION_DIR="${SAFETY_CURATION_DIR:-${NANOCHAT_BASE_DIR}/safety_response_curation}"

mkdir -p "$SAFETY_CURATION_DIR"

SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SEED="${SEED:-13}"

python3 "$ROOT_DIR/dev-safety/response_curation/step1_download_sample.py" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --output-dir "$SAFETY_CURATION_DIR"
