#!/bin/bash

# Monitor nanochat training progress
# Usage: ./monitor_training.sh [depth]

DEPTH=${1:-"20"}
CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints/d${DEPTH}"

echo "🔍 Monitoring nanochat d${DEPTH} training progress"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Function to get latest checkpoint info
get_latest_checkpoint() {
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "❌ No checkpoint directory found"
        return 1
    fi

    LATEST_MODEL=$(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_MODEL" ]; then
        echo "❌ No checkpoints found"
        return 1
    fi

    LATEST_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_0*//')
    echo "✅ Latest checkpoint: step $LATEST_STEP"
    
    # Show checkpoint details
    echo "   📁 File: $(basename $LATEST_MODEL)"
    echo "   📏 Size: $(du -h "$LATEST_MODEL" | cut -f1)"
    echo "   🕒 Created: $(stat -c %y "$LATEST_MODEL" 2>/dev/null | cut -d' ' -f1-2 || stat -f %Sm "$LATEST_MODEL" 2>/dev/null)"
    
    # Show training metadata if available
    META_FILE="$CHECKPOINT_DIR/meta_$(printf "%06d" $LATEST_STEP).json"
    if [ -f "$META_FILE" ] && command -v jq > /dev/null 2>&1; then
        echo "   📊 Progress:"
        jq -r '.step_count // "N/A"' "$META_FILE" 2>/dev/null | sed 's/^/      Steps: /'
        jq -r '.total_training_time // "N/A"' "$META_FILE" 2>/dev/null | sed 's/^/      Time: /' | sed 's/$/s/'
        jq -r '.val_bpb // "N/A"' "$META_FILE" 2>/dev/null | sed 's/^/      Val BPB: /'
    fi
    
    return 0
}

# Function to show running jobs
show_running_jobs() {
    echo ""
    echo "🏃 Running SLURM jobs:"
    squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %.15R" 2>/dev/null | grep -E "(JOBID|nanochat)" || echo "   No nanochat jobs running"
}

# Function to show recent log output
show_recent_logs() {
    echo ""
    echo "📋 Recent log output (last 20 lines):"
    
    # Find most recent output file
    RECENT_LOG=$(ls -1t ~/nanochat-*-*.out 2>/dev/null | head -1)
    if [ -n "$RECENT_LOG" ]; then
        echo "   📄 Log file: $(basename $RECENT_LOG)"
        echo "   ─────────────────────────────────────"
        tail -20 "$RECENT_LOG" 2>/dev/null | sed 's/^/   /'
        echo "   ─────────────────────────────────────"
    else
        echo "   No recent log files found"
    fi
}

# Function to estimate completion time
estimate_completion() {
    if [ ! -f "$META_FILE" ] || ! command -v jq > /dev/null 2>&1; then
        return 1
    fi
    
    local step_count=$(jq -r '.step_count // 0' "$META_FILE" 2>/dev/null)
    local training_time=$(jq -r '.total_training_time // 0' "$META_FILE" 2>/dev/null)
    local target_steps=$(jq -r '.num_iterations // 10000' "$META_FILE" 2>/dev/null)
    
    if [ "$step_count" -gt 0 ] && [ "$training_time" -gt 0 ] && [ "$target_steps" -gt 0 ]; then
        local steps_per_second=$(echo "scale=4; $step_count / $training_time" | bc -l 2>/dev/null || echo "0")
        local remaining_steps=$((target_steps - step_count))
        local estimated_remaining=$(echo "scale=0; $remaining_steps / $steps_per_second" | bc -l 2>/dev/null || echo "0")
        
        if [ "$estimated_remaining" -gt 0 ]; then
            echo ""
            echo "⏰ Estimated completion:"
            echo "   Progress: $step_count / $target_steps steps ($(echo "scale=1; $step_count * 100 / $target_steps" | bc -l)%)"
            echo "   Remaining: ~$(echo $estimated_remaining | awk '{printf "%.1f hours", $1/3600}')"
        fi
    fi
}

# Function to check storage usage
check_storage() {
    echo ""
    echo "💾 Storage usage:"
    du -sh ~ 2>/dev/null | awk '{print "   Home total: " $1}'
    du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "   nanochat cache: " $1}' || echo "   nanochat cache: 0"
    
    # Check if approaching limits
    HOME_USAGE=$(du -sb ~ 2>/dev/null | cut -f1)
    HOME_LIMIT=$((32 * 1024 * 1024 * 1024))  # 32GB limit
    if [ "$HOME_USAGE" -gt $((HOME_LIMIT * 80 / 100)) ]; then
        echo "   ⚠️  Warning: Home directory is >80% full"
    fi
}

# Main monitoring loop
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
get_latest_checkpoint
show_running_jobs
show_recent_logs
estimate_completion
check_storage

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Useful commands:"
echo "   Monitor logs: tail -f ~/nanochat-*.out"
echo "   Resume latest: ./runs/uppmax/resume_latest.sh $DEPTH"
echo "   Check jobs: squeue -u $USER"
echo "   Cancel job: scancel <job_id>"