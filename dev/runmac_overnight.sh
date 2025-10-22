#!/bin/bash
# Optimized overnight training for Mac (MPS/Apple Silicon)
# Expected runtime: 8-12 hours
# Expected result: Much better chatbot with coherent responses

set -e  # Exit on error

echo "=================================="
echo "nanochat Mac Overnight Training"
echo "=================================="
echo "Started: $(date)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Memory-based configuration
# Detect system memory (in GB) or allow manual override
if [ -z "$MEMORY_SIZE" ]; then
    MEMORY_SIZE=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
    echo "Auto-detected memory: ${MEMORY_SIZE}GB"
else
    echo "Using specified memory: ${MEMORY_SIZE}GB"
fi

# Calculate optimal batch sizes based on available memory
# Conservative estimates for MPS (unified memory shared with system)
# Note: total_batch_size must be divisible by (device_batch_size * max_seq_len)
# With max_seq_len=1024: device_batch_size * 1024 must divide total_batch_size
if [ $MEMORY_SIZE -ge 128 ]; then
    DEVICE_BATCH_SIZE=16
    TOTAL_BATCH_SIZE=16384    # 16 * 1024 = 16384
    EVAL_TOKENS=16384
    SPLIT_TOKENS=16384
    echo "Memory profile: 128GB+ (High performance)"
elif [ $MEMORY_SIZE -ge 64 ]; then
    DEVICE_BATCH_SIZE=8
    TOTAL_BATCH_SIZE=8192     # 8 * 1024 = 8192
    EVAL_TOKENS=8192
    SPLIT_TOKENS=8192
    echo "Memory profile: 64GB (Good performance)"
elif [ $MEMORY_SIZE -ge 32 ]; then
    DEVICE_BATCH_SIZE=4
    TOTAL_BATCH_SIZE=4096     # 4 * 1024 = 4096
    EVAL_TOKENS=4096
    SPLIT_TOKENS=4096
    echo "Memory profile: 32GB (Moderate performance)"
else
    DEVICE_BATCH_SIZE=1
    TOTAL_BATCH_SIZE=1024     # 1 * 1024 = 1024
    EVAL_TOKENS=2048
    SPLIT_TOKENS=2048
    echo "Memory profile: <32GB (Conservative)"
fi

# Allow manual overrides
DEPTH=${DEPTH:-6}                          # Bigger model (6 layers vs 4)
BASE_ITERATIONS=${BASE_ITERATIONS:-500}    # More base training
MID_ITERATIONS=${MID_ITERATIONS:-150}      # More midtraining
SFT_ITERATIONS=${SFT_ITERATIONS:-150}      # More SFT
DATA_SHARDS=${DATA_SHARDS:-50}             # More training data

echo ""
echo "Configuration:"
echo "  System Memory: ${MEMORY_SIZE}GB"
echo "  Model depth: $DEPTH (~82M params for d6)"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Total batch size: $TOTAL_BATCH_SIZE"
echo "  Eval tokens: $EVAL_TOKENS"
echo "  Base iterations: $BASE_ITERATIONS"
echo "  Mid iterations: $MID_ITERATIONS"
echo "  SFT iterations: $SFT_ITERATIONS"
echo "  Data shards: $DATA_SHARDS"
echo ""
echo "To override, set environment variables:"
echo "  MEMORY_SIZE=64 bash dev/runmac_overnight.sh"
echo "  DEVICE_BATCH_SIZE=8 bash dev/runmac_overnight.sh"
echo ""

# Clean up old run
echo "Cleaning up previous training..."
rm -f report.md
python -m nanochat.report reset

# Download training data
echo ""
echo "Step 1/6: Downloading training data ($DATA_SHARDS shards)..."
python -m nanochat.dataset -n $DATA_SHARDS

# Download identity conversations
echo ""
echo "Step 2/6: Downloading identity conversations..."
if [ ! -f ~/.cache/nanochat/identity_conversations.jsonl ]; then
    curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
      https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    echo "  Already downloaded, skipping."
fi

# Build tokenizer
echo ""
echo "Step 3/6: Training tokenizer..."
python -m nanochat.tokenizer

# Base model training
echo ""
echo "Step 4/6: Training base model ($BASE_ITERATIONS iterations)..."
echo "  Device batch size: $DEVICE_BATCH_SIZE, Total batch size: $TOTAL_BATCH_SIZE"
echo "  This will take ~2-4 hours..."
python -m scripts.base_train \
  --depth=$DEPTH \
  --max_seq_len=1024 \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --total_batch_size=$TOTAL_BATCH_SIZE \
  --num_iterations=$BASE_ITERATIONS \
  --eval_every=100 \
  --eval_tokens=$EVAL_TOKENS \
  --core_metric_every=250 \
  --core_metric_max_per_task=20 \
  --sample_every=100

# Evaluate base model
echo ""
echo "Evaluating base model..."
python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE --split_tokens=$SPLIT_TOKENS
python -m scripts.base_eval

# Midtraining
echo ""
echo "Step 5/6: Midtraining ($MID_ITERATIONS iterations)..."
echo "  Device batch size: $DEVICE_BATCH_SIZE, Total batch size: $TOTAL_BATCH_SIZE"
echo "  This will take ~2-3 hours..."
python -m scripts.mid_train \
  --num_iterations=$MID_ITERATIONS \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --max_seq_len=1024 \
  --total_batch_size=$TOTAL_BATCH_SIZE \
  --eval_every=50 \
  --eval_tokens=$EVAL_TOKENS

# SFT training
echo ""
echo "Step 6/6: Chat fine-tuning (SFT) ($SFT_ITERATIONS iterations)..."
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  This will take ~2-3 hours..."
python -m scripts.chat_sft \
  --num_iterations=$SFT_ITERATIONS \
  --device_batch_size=$DEVICE_BATCH_SIZE \
  --target_examples_per_step=$((DEVICE_BATCH_SIZE * 2)) \
  --eval_steps=10

# Final evaluation
echo ""
echo "Running final evaluations..."
python -m scripts.chat_eval -i sft || echo "Chat eval had issues, skipping..."

# Generate report
echo ""
echo "Generating final report..."
python -m nanochat.report generate

# Copy report to current directory
cp ~/.cache/nanochat/report/report.md ./report_overnight.md

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo "Finished: $(date)"
echo ""
echo "Your chatbot is ready! Chat with it:"
echo "  python -m scripts.chat_cli -i sft"
echo ""
echo "Or start the web UI:"
echo "  python -m scripts.chat_web -i sft"
echo ""
echo "Report saved to: report_overnight.md"
echo "=================================="
