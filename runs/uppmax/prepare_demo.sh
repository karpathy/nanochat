#!/bin/bash

# Prepare nanochat demo for Monday presentation
# Usage: ./prepare_demo.sh

echo "🎯 Preparing nanochat demo for Monday presentation..."

CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints/d20-demo"

# Check if we have a trained model
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ No demo model found at: $CHECKPOINT_DIR"
    echo "Start training with: sbatch runs/uppmax/train_d20_demo.sh"
    exit 1
fi

# Find latest checkpoint
LATEST_MODEL=$(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No model checkpoints found"
    exit 1
fi

LATEST_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_0*//')
echo "✅ Found model at step: $LATEST_STEP"

cd ~/nanochat
source .venv/bin/activate

echo ""
echo "=== Demo Model Performance ==="

# Get model metrics if available
META_FILE="$CHECKPOINT_DIR/meta_$(printf "%06d" $LATEST_STEP).json"
if [ -f "$META_FILE" ] && command -v jq > /dev/null 2>&1; then
    echo "📊 Training Metrics:"
    echo "   Steps completed: $(jq -r '.step_count // "N/A"' "$META_FILE")"
    echo "   Training time: $(jq -r '.total_training_time // "N/A"' "$META_FILE")s"
    echo "   Validation BPB: $(jq -r '.val_bpb // "N/A"' "$META_FILE")"
    echo "   CORE metric: $(jq -r '.core_metric // "N/A"' "$META_FILE")"
fi

echo ""
echo "=== Generating Demo Samples ==="

# Create demo samples for presentation
cat > demo_prompts.txt << 'EOF'
The future of artificial intelligence
Write a short story about a robot learning to paint
Explain quantum computing in simple terms
Create a recipe for chocolate chip cookies
Write a haiku about winter morning
EOF

echo "🤖 Generating samples for demo..."

# Generate samples for each prompt
while IFS= read -r prompt; do
    echo ""
    echo "Prompt: $prompt"
    echo "─────────────────────────────"
    python -m scripts.base_eval \
        --checkpoint-dir="$CHECKPOINT_DIR" \
        --step=$LATEST_STEP \
        --num-samples=1 \
        --prompt="$prompt" \
        --max-length=150 \
        --temperature=0.8 2>/dev/null | grep -A 20 "Generated samples:" | tail -n +2 || echo "Sample generation failed"
    echo ""
done < demo_prompts.txt

echo "=== Interactive Chat Demo ==="
echo "🎭 To start interactive demo:"
echo "   python -m scripts.chat_cli --checkpoint-dir='$CHECKPOINT_DIR' --step=$LATEST_STEP"
echo ""
echo "🌐 To start web demo:"
echo "   python -m scripts.chat_web --checkpoint-dir='$CHECKPOINT_DIR' --step=$LATEST_STEP"
echo "   Then visit: http://[UPPMAX_IP]:8000"

echo ""
echo "=== Demo Talking Points ==="
echo "📝 Key points for Monday presentation:"
echo "   • Trained a GPT-style model from scratch in ~24 hours"
echo "   • Model has $(jq -r '.model_config.n_layer // "20"' "$META_FILE" 2>/dev/null) layers, $(jq -r '.model_config.n_embd // "~1280"' "$META_FILE" 2>/dev/null) dimensions"
echo "   • Can generate coherent text on various topics"
echo "   • Demonstrates end-to-end LLM training pipeline"
echo "   • Cost: ~$72 for full GPT-2 capability (much less for d20)"
echo "   • Modular, hackable codebase perfect for research"

echo ""
echo "=== Files for Demo ==="
echo "📁 Important files:"
echo "   Model: $LATEST_MODEL"
echo "   Config: $META_FILE"
echo "   Logs: ~/nanochat-d20-demo-*.out"
echo "   Code: ~/nanochat/ (your fork with checkpoint improvements)"

echo ""
echo "🚀 DEMO READY! Good luck on Monday! 🎉"

# Cleanup
rm -f demo_prompts.txt