#!/bin/bash
set -euo pipefail

# Reproducible Chinese capability experiment for a local MPS/CPU machine.
# Run one stage at a time:
#   bash runs/runzh.sh prepare
#   bash runs/runzh.sh baseline
#   bash runs/runzh.sh sft
#   bash runs/runzh.sh cpt
#   bash runs/runzh.sh cpt-sft
#   bash runs/runzh.sh cpt-sft-lrfix
#   bash runs/runzh.sh eval-all

cd "$(dirname "$0")/.."
source .venv/bin/activate

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ZH_ROOT="${ZH_ROOT:-$NANOCHAT_BASE_DIR/zh_experiment}"
DEVICE_TYPE="${DEVICE_TYPE:-mps}"
STAGE="${1:-}"

if [ -z "$STAGE" ]; then
    echo "Usage: bash runs/runzh.sh prepare|baseline|sft|cpt|cpt-sft|cpt-sft-lrfix|eval-all"
    exit 1
fi

case "$STAGE" in
    prepare)
        python -m scripts.prepare_zh_experiment_data \
            --output-dir="$ZH_ROOT" \
            --pretrain-tokens=20000000 \
            --zh-token-ratio=0.70 \
            --sft-val-rows=1000
        ;;

    baseline)
        python -m scripts.zh_eval \
            -i sft -g d6 -s 1500 \
            --device-type="$DEVICE_TYPE" \
            --output="$ZH_ROOT/eval/sft_d6_001500.json"
        ;;

    sft)
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6 \
            --model-step=5000 \
            --output-model-tag=d6-zh-sft-demo \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --init-lr-frac=0.2 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=200 \
            --custom-train-jsonl="$ZH_ROOT/sft/train.jsonl" \
            --custom-val-jsonl="$ZH_ROOT/sft/val.jsonl" \
            --custom-train-token-ratio=0.30 \
            --run=dummy
        ;;

    cpt)
        python -m scripts.base_train \
            --device-type="$DEVICE_TYPE" \
            --depth=6 \
            --aspect-ratio=64 \
            --head-dim=64 \
            --window-pattern=L \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=1200 \
            --embedding-lr=0.03 \
            --unembedding-lr=0.0008 \
            --matrix-lr=0.002 \
            --scalar-lr=0.05 \
            --warmup-steps=40 \
            --warmdown-ratio=0.5 \
            --final-lr-frac=0.05 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --core-metric-every=-1 \
            --sample-every=200 \
            --save-every=200 \
            --init-from-model-tag=d6 \
            --init-from-step=5000 \
            --model-tag=d6-zh-cpt \
            --data-dir="$ZH_ROOT/pretrain" \
            --run=dummy
        ;;

    cpt-sft)
        # Historical low-LR run: chat_sft inherits the 0.1x CPT learning rates.
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-zh-cpt \
            --model-step=1200 \
            --output-model-tag=d6-zh-cpt-sft \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --init-lr-frac=0.2 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=200 \
            --custom-train-jsonl="$ZH_ROOT/sft/train.jsonl" \
            --custom-val-jsonl="$ZH_ROOT/sft/val.jsonl" \
            --custom-train-token-ratio=0.30 \
            --run=dummy
        ;;

    cpt-sft-lrfix)
        # Fair comparison with SFT-only: explicitly restore the original base LRs.
        python -m scripts.chat_sft \
            --device-type="$DEVICE_TYPE" \
            --model-tag=d6-zh-cpt \
            --model-step=1200 \
            --output-model-tag=d6-zh-cpt-sft-lrfix \
            --load-optimizer=0 \
            --max-seq-len=512 \
            --device-batch-size=32 \
            --total-batch-size=16384 \
            --num-iterations=600 \
            --embedding-lr=0.3 \
            --unembedding-lr=0.008 \
            --matrix-lr=0.02 \
            --init-lr-frac=0.2 \
            --eval-every=200 \
            --eval-tokens=131072 \
            --chatcore-every=-1 \
            --save-every=200 \
            --custom-train-jsonl="$ZH_ROOT/sft/train.jsonl" \
            --custom-val-jsonl="$ZH_ROOT/sft/val.jsonl" \
            --custom-train-token-ratio=0.30 \
            --run=dummy
        ;;

    eval-all)
        python -m scripts.zh_eval -i sft -g d6 -s 1500 --device-type="$DEVICE_TYPE"
        python -m scripts.zh_eval -i sft -g d6-zh-sft-demo -s 600 --device-type="$DEVICE_TYPE"
        python -m scripts.zh_eval -i sft -g d6-zh-cpt-sft-lrfix -s 600 --device-type="$DEVICE_TYPE"
        python -m scripts.base_eval \
            --eval=bpb --model-tag=d6 --step=5000 \
            --device-type="$DEVICE_TYPE" --device-batch-size=1 \
            --split-tokens=131072 --data-dir="$ZH_ROOT/pretrain_zh_eval"
        python -m scripts.base_eval \
            --eval=bpb --model-tag=d6-zh-cpt --step=1200 \
            --device-type="$DEVICE_TYPE" --device-batch-size=1 \
            --split-tokens=131072 --data-dir="$ZH_ROOT/pretrain_zh_eval"
        python -m scripts.chat_eval \
            -i sft -g d6-zh-sft-demo -s 600 -b 8 -x 128 \
            --device-type="$DEVICE_TYPE"
        python -m scripts.chat_eval \
            -i sft -g d6-zh-cpt-sft-lrfix -s 600 -b 8 -x 128 \
            --device-type="$DEVICE_TYPE"
        ;;

    *)
        echo "Unknown stage: $STAGE"
        exit 1
        ;;
esac
