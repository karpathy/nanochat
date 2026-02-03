#!/bin/bash
#
# IaC-GPT Complete Setup Script
#
# This script automates the entire IaC-GPT setup and training process:
# 1. Data scraping from GitHub
# 2. Data repackaging into training shards
# 3. Model training
# 4. Evaluation
#
# Usage:
#   export GITHUB_TOKEN="your_github_token"
#   bash runs/setup_iac_gpt.sh
#
# Expected time: ~4-5 hours on 8xH100 (including data collection)
# Expected cost: ~$100

set -e

echo "========================================"
echo "IaC-GPT Complete Setup & Training"
echo "========================================"
echo ""

# Configuration
MAX_REPOS=500
DATA_RAW_DIR="data/iac_raw"
DATA_SHARDS_DIR="$HOME/.cache/nanochat/iac_data"

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# Check Python environment
if ! python3 --version &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python $(python3 --version) found"

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "WARNING: GITHUB_TOKEN not set."
    echo "You'll be limited to 60 API requests/hour instead of 5000/hour."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please set GITHUB_TOKEN and try again:"
        echo "  export GITHUB_TOKEN='ghp_your_token_here'"
        exit 1
    fi
fi

# Check GPU availability
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: No CUDA GPUs detected!"
    echo "This script is designed for 8xH100 GPU training."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Data Collection
echo ""
echo "========================================"
echo "Step 1: Collecting IaC Training Data"
echo "========================================"
echo ""
echo "This will scrape $MAX_REPOS high-quality repositories from GitHub."
echo "Expected time: 20-40 minutes"
echo ""

if [ -d "$DATA_RAW_DIR/terraform" ] && [ "$(ls -A $DATA_RAW_DIR/terraform 2>/dev/null)" ]; then
    echo "Found existing data in $DATA_RAW_DIR"
    read -p "Skip data collection and use existing data? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Scraping data..."
        python3 dev/scrape_iac_data.py \
            --output-dir "$DATA_RAW_DIR" \
            --max-repos "$MAX_REPOS"
    else
        echo "Using existing data."
    fi
else
    echo "Scraping data..."
    python3 dev/scrape_iac_data.py \
        --output-dir "$DATA_RAW_DIR" \
        --max-repos "$MAX_REPOS"
fi

# Step 2: Data Repackaging
echo ""
echo "========================================"
echo "Step 2: Repackaging Data into Shards"
echo "========================================"
echo ""
echo "Converting raw IaC files to parquet shards..."
echo ""

python3 dev/repackage_iac_data.py \
    --input-dir "$DATA_RAW_DIR" \
    --output-dir "$DATA_SHARDS_DIR" \
    --include-synthetic \
    --include-docs

# Check if shards were created
SHARD_COUNT=$(ls -1 $DATA_SHARDS_DIR/shard_*.parquet 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No shards created! Please check the repackaging step."
    exit 1
fi

echo ""
echo "✅ Created $SHARD_COUNT data shards"

# Step 3: Training
echo ""
echo "========================================"
echo "Step 3: Training IaC-GPT Model"
echo "========================================"
echo ""
echo "Starting speedrun training on 8 GPUs..."
echo "Expected time: 3-4 hours"
echo "Expected cost: ~$75 on 8xH100 @ $24/hour"
echo ""
echo "💡 Tip: Run this in a screen/tmux session!"
echo ""

read -p "Ready to start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping training. You can run it later with:"
    echo "  bash runs/speedrun_iac.sh"
    exit 0
fi

# Run the training script
bash runs/speedrun_iac.sh

# Step 4: Evaluation
echo ""
echo "========================================"
echo "Step 4: Evaluating Model"
echo "========================================"
echo ""

# Find the latest model
LATEST_MODEL=$(ls -td logs/iac_gpt_* 2>/dev/null | head -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "ERROR: No trained model found!"
    exit 1
fi

echo "Evaluating model: $LATEST_MODEL"
echo ""

python3 -m scripts.base_eval \
    --model "$LATEST_MODEL/latest_checkpoint"

# Final Summary
echo ""
echo "========================================"
echo "🎉 IaC-GPT Setup Complete!"
echo "========================================"
echo ""
echo "Your IaC specialist model is ready:"
echo "  Model: $LATEST_MODEL"
echo ""
echo "Try it out:"
echo "  # Web UI (recommended)"
echo "  python -m scripts.chat_web --model $LATEST_MODEL/latest_checkpoint"
echo ""
echo "  # Command-line interface"
echo "  python -m scripts.chat_cli --model $LATEST_MODEL/latest_checkpoint"
echo ""
echo "  # Specialized IaC CLI"
echo "  python scripts/iac_cli.py generate --type terraform --service eks"
echo "  python scripts/iac_cli.py audit --path infrastructure/"
echo "  python scripts/iac_cli.py interactive"
echo ""
echo "Happy Infrastructure Coding! 🚀"
