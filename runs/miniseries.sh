#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
cargo build --release --features "${RUST_AGENT_FEATURES:-cuda}"
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
"$RUST_AGENT" download-dataset --dataset climbmix --num-files "${CLIMBMIX_FILES:-1000}"
"$RUST_AGENT" train-tokenizer --dataset climbmix --max-chars 2000000000
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"
DEPTHS="${DEPTHS:-12 14 16 18 20 22 24 26}"

for depth in $DEPTHS; do
  tag="${SERIES_NAME}_miniseries_d${depth}"
  batch_size="${DEVICE_BATCH_SIZE:-32}"
  if (( depth >= 20 )); then batch_size="${LARGE_BATCH_SIZE:-16}"; fi
  "$RUST_AGENT" train --depth "$depth" --model-tag "$tag" \
    --run "${WANDB_RUN}_d${depth}" --batch-size "$batch_size" \
    --core-metric-every 999999 --core-metric-max-per-task -1 \
    --sample-every -1 --save-every -1 --device auto
done
