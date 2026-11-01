#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
cargo build --release --features "${RUST_AGENT_FEATURES:-cuda}"
RUST_AGENT="$REPO_ROOT/target/release/rust-agent"
LABEL="${LABEL:-rust}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
DEPTHS="${DEPTHS:-10 12 14 16 18 20}"
FLOPS_BUDGETS="${FLOPS_BUDGETS:-1e18 2.15e18 4.64e18 1e19}"

for budget in $FLOPS_BUDGETS; do
  for depth in $DEPTHS; do
    tag="${LABEL}_flops${budget}_d${depth}"
    "$RUST_AGENT" train --depth "$depth" --target-flops "$budget" \
      --model-tag "$tag" --run "${WANDB_RUN}_d${depth}_flops${budget}" \
      --core-metric-every 999999 --core-metric-max-per-task -1 \
      --sample-every -1 --save-every -1 --device auto
  done
done
