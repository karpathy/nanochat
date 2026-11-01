#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$REPO_ROOT/output}"
mkdir -p "$NANOCHAT_BASE_DIR" "$REPO_ROOT/logs"
exec > >(tee "$REPO_ROOT/logs/wsl-run.log") 2>&1

cargo build --release --features "${RUST_AGENT_FEATURES:-cuda}"
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
WANDB_RUN="${WANDB_RUN:-dummy}"

"$RUST_AGENT" download-dataset --dataset crowdmind_2048
"$RUST_AGENT" download-dataset --dataset climbmix --num-files "${CLIMBMIX_FILES:-170}"
"$RUST_AGENT" train-tokenizer --dataset climbmix

"$RUST_AGENT" train --depth 12 --max-seq-len 16384 \
  --target-param-data-ratio 8 --batch-size 1 --save-every 100 \
  --dataset climbmix --model-tag d12 --run "$WANDB_RUN" --device auto
"$RUST_AGENT" base-eval --model-tag d12 --device-batch-size 2 --device auto

"$RUST_AGENT" train --source base --model-tag d12 --output-model-tag d12-smoltalk \
  --data-mode standard --max-seq-len 16384 --batch-size 2 --run "$WANDB_RUN" --device auto
"$RUST_AGENT" train --source sft --model-tag d12-smoltalk --output-model-tag d12-tools \
  --data-mode tools --load-optimizer 0 --max-seq-len 16384 --batch-size 2 --run "$WANDB_RUN" --device auto
"$RUST_AGENT" train --source sft --model-tag d12-tools --output-model-tag d12-post-training \
  --data-mode post-training --load-optimizer 0 --max-seq-len 16384 --batch-size 2 --run "$WANDB_RUN" --device auto
"$RUST_AGENT" train --source sft --model-tag d12-post-training --output-model-tag d12-traces \
  --data-mode traces --load-optimizer 0 --max-seq-len 16384 --batch-size 2 --run "$WANDB_RUN" --device auto
"$RUST_AGENT" eval-chat --source sft --model-tag d12-traces --device auto
