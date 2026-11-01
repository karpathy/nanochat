#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

RUST_AGENT_FEATURES="${RUST_AGENT_FEATURES:-cuda,cuda-nccl}"
cargo build --release --features "$RUST_AGENT_FEATURES"
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

"$RUST_AGENT" download-dataset --dataset crowdmind_2048
"$RUST_AGENT" download-dataset --dataset climbmix --num-files "${CLIMBMIX_FILES:-170}"
"$RUST_AGENT" train-tokenizer --dataset climbmix

"$RUST_AGENT" launch --nproc-per-node "$NPROC_PER_NODE" train \
  --depth 24 --target-param-data-ratio 8 --batch-size "$BATCH_SIZE" \
  --fp8 --run "$WANDB_RUN" --model-tag d24
"$RUST_AGENT" base-eval --model-tag d24 --device-batch-size "$BATCH_SIZE"
"$RUST_AGENT" launch --nproc-per-node "$NPROC_PER_NODE" train \
  --source base --model-tag d24 --output-model-tag d24-sft \
  --data-mode standard --batch-size "$BATCH_SIZE" --run "$WANDB_RUN"
"$RUST_AGENT" eval-chat --source sft --model-tag d24-sft
