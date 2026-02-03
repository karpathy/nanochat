#!/bin/bash
#
# IaC-GPT Speedrun Training Script
# 
# This script trains a GPT-2 grade model optimized for Infrastructure-as-Code
# in approximately 3-4 hours on an 8xH100 GPU node.
#
# Prerequisites:
# 1. Run: python dev/scrape_iac_data.py --output-dir data/iac_raw --max-repos 500
# 2. Run: python dev/repackage_iac_data.py --input-dir data/iac_raw --output-dir ~/.cache/nanochat/iac_data --include-synthetic --include-docs
#
# Usage:
#   bash runs/speedrun_iac.sh
#
# Expected cost: ~$75 on 8xH100 @ $24/hour
# Expected time: ~3 hours

set -e

echo "========================================"
echo "IaC-GPT Training Pipeline"
echo "========================================"
echo ""

# Configuration
RUN_NAME="iac_gpt_d24_$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="iac_gpt_d24"
DEPTH=24
DEVICE_BATCH_SIZE=16
DATA_DIR="$HOME/.cache/nanochat/iac_data"

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    echo "Please run the data preparation scripts first:"
    echo "  1. python dev/scrape_iac_data.py --output-dir data/iac_raw --max-repos 500"
    echo "  2. python dev/repackage_iac_data.py --input-dir data/iac_raw --output-dir ~/.cache/nanochat/iac_data --include-synthetic --include-docs"
    exit 1
fi

# Count shards
SHARD_COUNT=$(ls -1 $DATA_DIR/shard_*.parquet 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No data shards found in $DATA_DIR"
    echo "Please run: python dev/repackage_iac_data.py"
    exit 1
fi

echo "Configuration:"
echo "  Run name: $RUN_NAME"
echo "  Model depth: $DEPTH layers"
echo "  Data directory: $DATA_DIR"
echo "  Shards found: $SHARD_COUNT"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo ""

# Step 1: Pretrain the base model
echo "========================================"
echo "Step 1: Pretraining IaC-GPT Base Model"
echo "========================================"
echo ""

# Override the DATA_DIR in dataset.py by creating a symlink
mkdir -p ~/.cache/nanochat/
if [ -L ~/.cache/nanochat/base_data ]; then
    rm ~/.cache/nanochat/base_data
fi
ln -s "$DATA_DIR" ~/.cache/nanochat/base_data

echo "Starting distributed training on 8 GPUs..."
echo ""

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=$DEPTH \
    --run="$RUN_NAME" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --sample-every=3000 \
    --save-every=3000 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12

echo ""
echo "========================================"
echo "Pretraining Complete!"
echo "========================================"
echo ""

# Step 2: Identity Infusion (Optional - makes it a "Senior DevOps Architect")
echo "========================================"
echo "Step 2: Identity Infusion (Optional)"
echo "========================================"
echo ""

read -p "Would you like to add 'Senior DevOps Architect' persona? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating synthetic identity data..."
    python dev/gen_iac_identity.py --output data/iac_identity.jsonl
    
    echo "Fine-tuning with identity..."
    OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
        --base-model="logs/$RUN_NAME/latest_checkpoint" \
        --data="data/iac_identity.jsonl" \
        --model-tag="${MODEL_TAG}_identity" \
        --run="${RUN_NAME}_identity" \
        --device-batch-size=8 \
        --max-steps=1000
    
    FINAL_MODEL="logs/${RUN_NAME}_identity/latest_checkpoint"
else
    echo "Skipping identity infusion."
    FINAL_MODEL="logs/$RUN_NAME/latest_checkpoint"
fi

echo ""
echo "========================================"
echo "Training Pipeline Complete!"
echo "========================================"
echo ""
echo "Your IaC-GPT model is ready at:"
echo "  $FINAL_MODEL"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python -m scripts.base_eval --model $FINAL_MODEL"
echo "  2. Chat (Web): python -m scripts.chat_web --model $FINAL_MODEL"
echo "  3. Chat (CLI): python -m scripts.chat_cli --model $FINAL_MODEL"
echo "  4. IaC CLI: python scripts/iac_cli.py --model $FINAL_MODEL"
echo ""
echo "Happy Infrastructure Coding! 🚀"
