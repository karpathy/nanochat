#!/usr/bin/env bash
set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_pg"
export NANOCHAT_TOKENIZER_KIND=sentencepiece

PYTHONPATH=. python -m scripts.base_train \
  --dataset-kind paramgolf \
  --pg-data-path ../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --depth 4 \
  --max-seq-len 128 \
  --window-pattern L \
  --device-batch-size 2 \
  --total-batch-size 1024 \
  --num-iterations 50 \
  --eval-every 25 \
  --core-metric-every -1 \
  --sample-every -1 \
  --save-every -1 \
  --run dummy
