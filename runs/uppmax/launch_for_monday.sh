#!/bin/bash

# Launch nanochat training for Monday demo
# This script ensures training finishes by Monday 11 AM

echo "🎯 Launching nanochat d20 training for Monday 11 AM demo"

# Calculate timing
CURRENT_TIME=$(date +%s)
MONDAY_11AM=$(date -d "next Monday 11:00" +%s)
HOURS_UNTIL_DEMO=$(( (MONDAY_11AM - CURRENT_TIME) / 3600 ))

echo "📅 Current time: $(date)"
echo "🎯 Target demo time: $(date -d "next Monday 11:00")"
echo "⏰ Hours until demo: $HOURS_UNTIL_DEMO"

# Check if we have enough time
if [ $HOURS_UNTIL_DEMO -lt 25 ]; then
    echo "⚠️  WARNING: Less than 25 hours until demo!"
    echo "   Training needs ~24 hours to complete"
    echo "   Consider starting immediately"
fi

# Check for existing training
if squeue -u $USER | grep -q nanochat; then
    echo "🏃 Existing nanochat job found:"
    squeue -u $USER | grep nanochat
    echo ""
    echo "Options:"
    echo "  1. Let current job finish"
    echo "  2. Cancel and start demo-optimized training: scancel <job_id>"
    exit 0
fi

# Check for existing demo model
DEMO_DIR="$HOME/.cache/nanochat/base_checkpoints/d20-demo"
if [ -d "$DEMO_DIR" ] && [ -n "$(ls -A "$DEMO_DIR" 2>/dev/null)" ]; then
    echo "✅ Existing demo model found!"
    echo "📁 Location: $DEMO_DIR"
    
    # Check if it's complete enough for demo
    LATEST_MODEL=$(ls -1 "$DEMO_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_MODEL" ]; then
        LATEST_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_0*//')
        echo "📊 Latest checkpoint: step $LATEST_STEP"
        
        if [ $LATEST_STEP -gt 2000 ]; then
            echo "🎉 Model looks ready for demo!"
            echo "Run: ./runs/uppmax/prepare_demo.sh"
            exit 0
        else
            echo "⚠️  Model needs more training (step $LATEST_STEP < 2000)"
        fi
    fi
fi

echo ""
echo "🚀 Starting demo-optimized d20 training..."
echo "⏱️  Training time: ~24 hours"
echo "💾 Checkpoints: Every 1.5 hours"
echo "🎯 Optimized for Monday demo"

# Submit the demo job
JOB_ID=$(sbatch runs/uppmax/train_d20_demo.sh | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully!"
    echo "📋 Job ID: $JOB_ID"
    echo "📄 Monitor logs: tail -f ~/nanochat-d20-demo-$JOB_ID.out"
    echo "📊 Check status: squeue -u $USER"
    echo ""
    echo "⏰ Expected completion: $(date -d "+24 hours")"
    echo "🎯 Ready for demo: Monday $(date -d "next Monday 11:00" '+%H:%M')"
    echo ""
    echo "💡 Useful commands:"
    echo "   Monitor: ./runs/uppmax/monitor_training.sh"
    echo "   Resume if needed: ./runs/uppmax/resume_latest.sh 20"
    echo "   Prepare demo: ./runs/uppmax/prepare_demo.sh"
else
    echo "❌ Failed to submit job"
    echo "Check SLURM status: sinfo"
    exit 1
fi

echo ""
echo "🎉 All set for Monday! Training will complete in time for your 11 AM meeting."