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

# Configuration
DEPTH=6                    # Bigger model (6 layers vs 4)
BASE_ITERATIONS=500        # More base training
MID_ITERATIONS=150         # More midtraining
SFT_ITERATIONS=150         # More SFT
DATA_SHARDS=50             # More training data

echo "Configuration:"
echo "  Model depth: $DEPTH (36.7M → 82M params)"
echo "  Base iterations: $BASE_ITERATIONS"
echo "  Mid iterations: $MID_ITERATIONS"
echo "  SFT iterations: $SFT_ITERATIONS"
echo "  Data shards: $DATA_SHARDS"
echo ""

# Clean up old run
echo "Cleaning up previous training..."
rm -f report.md
python -m scripts.report --reset

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
echo "  This will take ~2-4 hours..."
python -m scripts.base_train \
  --depth=$DEPTH \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --num_iterations=$BASE_ITERATIONS \
  --eval_every=100 \
  --eval_tokens=8192 \
  --core_metric_every=250 \
  --core_metric_max_per_task=20 \
  --sample_every=100

# Evaluate base model
echo ""
echo "Evaluating base model..."
python -m scripts.base_loss
python -m scripts.base_eval

# Midtraining
echo ""
echo "Step 5/6: Midtraining ($MID_ITERATIONS iterations)..."
echo "  This will take ~2-3 hours..."
python -m scripts.mid_train \
  --num_iterations=$MID_ITERATIONS \
  --device_batch_size=1 \
  --max_seq_len=1024 \
  --total_batch_size=1024 \
  --eval_every=50

# SFT training
echo ""
echo "Step 6/6: Chat fine-tuning (SFT) ($SFT_ITERATIONS iterations)..."
echo "  This will take ~2-3 hours..."
python -m scripts.chat_sft \
  --num_iterations=$SFT_ITERATIONS \
  --device_batch_size=1 \
  --target_examples_per_step=8 \
  --eval_steps=10

# Final evaluation
echo ""
echo "Running final evaluations..."
python -m scripts.chat_eval -i sft || echo "Chat eval had issues, skipping..."

# Generate report
echo ""
echo "Generating final report..."
python -m scripts.report

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
