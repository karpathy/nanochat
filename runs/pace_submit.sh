#!/bin/bash

# Pipeline:
#   Stage 1  — CPU: tokenizer + dataset
#   Stage 2a — GPU: pretraining chunk 1
#   Stage 2b — GPU: auto-resume chunk 2
#   Stage 2c — GPU: auto-resume chunk 3
#   Stage 2d — GPU: auto-resume chunk 4
#   Stage 3  — GPU: base eval + SFT + chat eval + report
#
# Usage (from repo root):
#   bash runs/pace_submit.sh
#
# Optional W&B logging:
#   WANDB_RUN=my-run bash runs/pace_submit.sh

set -e
cd "$HOME/scratch/nanochat"

mkdir -p runs/logs

WANDB_RUN="${WANDB_RUN:-dummy}"
export WANDB_RUN

echo "Submitting nanochat full pipeline..."
echo "WANDB_RUN=$WANDB_RUN"
echo ""

# Stage 1
JOB1=$(sbatch --parsable \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage1_tokenizer.sh)
echo "Stage 1 submitted: job $JOB1 (tokenizer + dataset)"

# Stage 2a
JOB2A=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage2a_pretrain.sh)
echo "Stage 2a submitted: job $JOB2A (pretrain chunk 1, depends on $JOB1)"

# Stage 2b
JOB2B=$(sbatch --parsable \
    --dependency=afterany:$JOB2A \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage2b_pretrain.sh)
echo "Stage 2b submitted: job $JOB2B (pretrain chunk 2, depends on $JOB2A)"

# Stage 2c
JOB2C=$(sbatch --parsable \
    --dependency=afterany:$JOB2B \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage2c_pretrain.sh)
echo "Stage 2c submitted: job $JOB2C (pretrain chunk 3, depends on $JOB2B)"

# Stage 2d
JOB2D=$(sbatch --parsable \
    --dependency=afterany:$JOB2C \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage2d_pretrain.sh)
echo "Stage 2d submitted: job $JOB2D (pretrain chunk 4, depends on $JOB2C)"

# Stage 3
JOB3=$(sbatch --parsable \
    --dependency=afterok:$JOB2D \
    --export=ALL,WANDB_RUN=$WANDB_RUN \
    runs/pace_stage3_sft.sh)
echo "Stage 3 submitted:  job $JOB3  (eval + SFT, depends on $JOB2D)"

echo ""
echo "All jobs queued. Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f runs/logs/stage1_${JOB1}.out"
echo "  tail -f runs/logs/stage2a_${JOB2A}.out"
echo "  tail -f runs/logs/stage2b_${JOB2B}.out"
echo "  tail -f runs/logs/stage2c_${JOB2C}.out"
echo "  tail -f runs/logs/stage2d_${JOB2D}.out"
echo "  tail -f runs/logs/stage3_${JOB3}.out"
echo ""
echo "To cancel everything:"
echo "  scancel $JOB1 $JOB2A $JOB2B $JOB2C $JOB2D $JOB3"
