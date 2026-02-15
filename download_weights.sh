#!/bin/bash
# download_weights.sh - Download nanochat weights from UPPMAX to local machine

set -e

# Configuration
UPPMAX_USER="birger"
UPPMAX_HOST="pelle.uppmax.uu.se"
LOCAL_DIR="checkpoints/nanochat-d20"

echo "🔽 Downloading nanochat weights from UPPMAX..."

# Create local directory
mkdir -p "$LOCAL_DIR"

# Download all checkpoints from the d20 run
echo "📁 Downloading checkpoints..."
scp -r "${UPPMAX_USER}@${UPPMAX_HOST}:~/.cache/nanochat/base_checkpoints/d20/*" "$LOCAL_DIR/"

# Download training logs (find the most recent)
echo "📋 Downloading training logs..."
scp "${UPPMAX_USER}@${UPPMAX_HOST}:~/nanochat-*.out" "$LOCAL_DIR/" 2>/dev/null || echo "⚠️  No training logs found"

# List what we downloaded
echo "✅ Download complete! Files in $LOCAL_DIR:"
ls -la "$LOCAL_DIR"

echo "🎯 Best checkpoint is likely: $LOCAL_DIR/step_2500.pt"
echo "📊 Check training progress in the .out log files"