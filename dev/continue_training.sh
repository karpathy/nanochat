#!/bin/bash
# Smart training continuation script
# Checks for existing checkpoints and continues from where you left off

set -e

echo "=================================="
echo "nanochat Training Continuation"
echo "=================================="
echo "Started: $(date)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Memory-based configuration (same as runmac_overnight.sh)
if [ -z "$MEMORY_SIZE" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_SIZE=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
        echo "Auto-detected memory: ${MEMORY_SIZE}GB"
    else
        MEMORY_SIZE=16
    fi
fi

# Calculate optimal batch sizes
if [ $MEMORY_SIZE -ge 128 ]; then
    DEVICE_BATCH_SIZE=16
    TOTAL_BATCH_SIZE=16384
    EVAL_TOKENS=16384
    SPLIT_TOKENS=16384
elif [ $MEMORY_SIZE -ge 64 ]; then
    DEVICE_BATCH_SIZE=8
    TOTAL_BATCH_SIZE=8192
    EVAL_TOKENS=8192
    SPLIT_TOKENS=8192
elif [ $MEMORY_SIZE -ge 32 ]; then
    DEVICE_BATCH_SIZE=4
    TOTAL_BATCH_SIZE=4096
    EVAL_TOKENS=4096
    SPLIT_TOKENS=4096
else
    DEVICE_BATCH_SIZE=1
    TOTAL_BATCH_SIZE=1024
    EVAL_TOKENS=2048
    SPLIT_TOKENS=2048
fi

# Allow manual overrides
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
MID_ITERATIONS=${MID_ITERATIONS:-150}
SFT_ITERATIONS=${SFT_ITERATIONS:-150}

echo "Configuration:"
echo "  Memory: ${MEMORY_SIZE}GB"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Total batch size: $TOTAL_BATCH_SIZE"
echo ""

# Check what exists
CACHE_DIR="$HOME/.cache/nanochat"
BASE_DIR="$CACHE_DIR/base_checkpoints"
MID_DIR="$CACHE_DIR/mid_checkpoints"
SFT_DIR="$CACHE_DIR/sft_checkpoints"

echo "Checking existing checkpoints..."
echo ""

# Function to find latest checkpoint and extract tag
find_latest_checkpoint() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "none"
        return
    fi
    # Find the latest model tag directory
    local latest_tag=$(ls -1 "$dir" 2>/dev/null | grep -E "^d[0-9]+$" | sort -V | tail -1)
    if [ -z "$latest_tag" ]; then
        echo "none"
        return
    fi
    # Find the latest step in that tag
    local latest_step=$(ls -1 "$dir/$latest_tag" 2>/dev/null | grep -E "^model_[0-9]+\.pt$" | sed 's/model_//;s/\.pt//' | sort -n | tail -1)
    if [ -z "$latest_step" ]; then
        echo "none"
        return
    fi
    echo "$latest_tag/step_$latest_step"
}

BASE_CHECKPOINT=$(find_latest_checkpoint "$BASE_DIR")
MID_CHECKPOINT=$(find_latest_checkpoint "$MID_DIR")
SFT_CHECKPOINT=$(find_latest_checkpoint "$SFT_DIR")

# Extract base model tag (e.g., "d8" from "d8/step_001000")
BASE_TAG=$(echo $BASE_CHECKPOINT | cut -d'/' -f1)
MID_TAG=$(echo $MID_CHECKPOINT | cut -d'/' -f1)
SFT_TAG=$(echo $SFT_CHECKPOINT | cut -d'/' -f1)

echo "Status:"
if [ "$BASE_CHECKPOINT" != "none" ]; then
    echo "  ✓ Base model: $BASE_CHECKPOINT"
else
    echo "  ✗ Base model: Not found"
fi

if [ "$MID_CHECKPOINT" != "none" ]; then
    echo "  ✓ Midtraining: $MID_CHECKPOINT"
else
    echo "  ✗ Midtraining: Not found"
fi

if [ "$SFT_CHECKPOINT" != "none" ]; then
    echo "  ✓ SFT: $SFT_CHECKPOINT"
else
    echo "  ✗ SFT: Not found"
fi
echo ""

# Determine what to do
if [ "$SFT_CHECKPOINT" != "none" ]; then
    echo "🎉 All training stages complete!"
    echo ""
    echo "Your chatbot is ready. Chat with:"
    echo "  python -m scripts.chat_cli -i sft"
    echo ""
    echo "Or start web UI:"
    echo "  python -m scripts.chat_web -i sft"
    echo ""
    exit 0
fi

if [ "$BASE_CHECKPOINT" = "none" ]; then
    echo "❌ No base model found. Please run base training first:"
    echo "  bash dev/runmac_overnight.sh"
    echo ""
    exit 1
fi

# Download identity conversations if needed
if [ ! -f "$CACHE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity conversations..."
    curl -L -o "$CACHE_DIR/identity_conversations.jsonl" \
      https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    echo ""
fi

# Continue from where we left off
# Check if we need midtraining for the current base model tag
if [ "$MID_CHECKPOINT" = "none" ] || [ "$MID_TAG" != "$BASE_TAG" ]; then
    if [ "$MID_TAG" != "$BASE_TAG" ] && [ "$MID_CHECKPOINT" != "none" ]; then
        echo "⚠️  Found mid checkpoint for $MID_TAG but base model is $BASE_TAG"
        echo "    Need to run midtraining for $BASE_TAG"
    fi

    echo "📍 Continuing from: Base model complete ($BASE_TAG)"
    echo "📋 Next steps: Midtraining → SFT"
    echo ""

    # Run midtraining
    echo "Step 1/2: Midtraining ($MID_ITERATIONS iterations)..."
    echo "  Loading base checkpoint: $BASE_CHECKPOINT"
    echo "  Device batch size: $DEVICE_BATCH_SIZE"
    python -m scripts.mid_train \
      --num_iterations=$MID_ITERATIONS \
      --device_batch_size=$DEVICE_BATCH_SIZE \
      --max_seq_len=1024 \
      --total_batch_size=$TOTAL_BATCH_SIZE \
      --eval_every=50 \
      --eval_tokens=$EVAL_TOKENS

    echo ""
    echo "✓ Midtraining complete!"
    echo ""
fi

# Check again for mid checkpoint and verify tag matches
MID_CHECKPOINT=$(find_latest_checkpoint "$MID_DIR")
MID_TAG=$(echo $MID_CHECKPOINT | cut -d'/' -f1)

if [ "$MID_CHECKPOINT" = "none" ]; then
    echo "❌ Midtraining failed to produce checkpoint"
    exit 1
fi

# Verify tags match
if [ "$MID_TAG" != "$BASE_TAG" ]; then
    echo "❌ Tag mismatch: Base is $BASE_TAG but mid is $MID_TAG"
    echo "This shouldn't happen. Please check checkpoints manually."
    exit 1
fi

# Check if we need SFT for the current mid model tag
if [ "$SFT_CHECKPOINT" = "none" ] || [ "$SFT_TAG" != "$MID_TAG" ]; then
    if [ "$SFT_TAG" != "$MID_TAG" ] && [ "$SFT_CHECKPOINT" != "none" ]; then
        echo "⚠️  Found SFT checkpoint for $SFT_TAG but mid model is $MID_TAG"
        echo "    Need to run SFT for $MID_TAG"
    fi

    # Run SFT
    echo "📍 Continuing from: Midtraining complete ($MID_TAG)"
    echo "📋 Next step: SFT (final stage!)"
    echo ""
    echo "Step 2/2: Chat fine-tuning (SFT) ($SFT_ITERATIONS iterations)..."
    echo "  Loading mid checkpoint: $MID_CHECKPOINT"
    echo "  Device batch size: $DEVICE_BATCH_SIZE"
    python -m scripts.chat_sft \
      --num_iterations=$SFT_ITERATIONS \
      --device_batch_size=$DEVICE_BATCH_SIZE \
      --target_examples_per_step=$((DEVICE_BATCH_SIZE * 2)) \
      --eval_steps=10
else
    echo "✓ SFT already complete for $SFT_TAG"
fi

echo ""
echo "=================================="
echo "🎉 All Training Complete!"
echo "=================================="
echo "Finished: $(date)"
echo ""
echo "Your chatbot is ready! Chat with:"
echo "  python -m scripts.chat_cli -i sft"
echo ""
echo "Or start the web UI:"
echo "  python -m scripts.chat_web -i sft"
echo ""
echo "Generate final report:"
echo "  python -m nanochat.report generate"
echo "=================================="
