#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

cargo build --release
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
WANDB_RUN="${WANDB_RUN:-dummy}"
"$RUST_AGENT" download-dataset --dataset climbmix --num-files "${CLIMBMIX_FILES:-8}"
"$RUST_AGENT" train-tokenizer --dataset climbmix --max-chars "${MAX_CHARS:-2000000000}"
"$RUST_AGENT" train --depth 6 --head-dim 64 --window-pattern L \
  --max-seq-len 512 --batch-size 32 --total-batch-size 16384 \
  --eval-every 100 --eval-tokens 524288 --core-metric-every -1 \
  --sample-every 100 --steps 5000 --model-tag d6 \
  --run "$WANDB_RUN" --device cpu
"$RUST_AGENT" base-eval --model-tag d6 --device-batch-size 1 \
  --split-tokens 16384 --max-per-task 16 --device cpu
"$RUST_AGENT" train --source base --model-tag d6 --output-model-tag d6-sft \
  --data-mode standard --steps 1500 --eval-every 200 \
  --eval-tokens 524288 --run "$WANDB_RUN" --device cpu
