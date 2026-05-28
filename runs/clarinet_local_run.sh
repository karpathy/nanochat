#!/bin/bash
# clarinet local run for single-GPU setups (e.g. AMD RX 7900 XT in WSL2 with ROCm).
#
# Differences from runs/clarinet_speedrun.sh (which targets 8xH100 + Hopper-specific features):
#   - no torchrun / single GPU
#   - no --fp8 (Hopper-only)
#   - much smaller --device-batch-size (gradient accumulation makes up the effective batch)
#   - bfloat16 dtype explicitly set
#   - --save-every=500 so the run is resumable if power flickers
#   - trimmed IV sweep (4 weights, 3 tasks) to fit a ~14-day local budget
#
# Expected wall-clock at d24 on a single RX 7900 XT (~70 TFLOPs achieved bf16, no FA3):
#   pretraining: ~12-15 days, SFT: ~4-6 hrs, chat_eval: ~6-12 hrs, IV sweep: ~12-24 hrs
#   total: ~13-17 days
#
# Launch unattended so it survives terminal close and WSL idle:
#   mkdir -p ~/clarinet/logs
#   screen -L -Logfile ~/clarinet/logs/run.log -dmS clarinet \
#       bash -c "nohup bash runs/clarinet_local_run.sh > ~/clarinet/logs/run.full.log 2>&1"
#   screen -ls    # verify 'clarinet' detached
#
# Resume after a crash (checkpoint dir name printed at run start):
#   bash runs/clarinet_local_run.sh    # the train scripts auto-resume from latest

set -e

export OMP_NUM_THREADS=4
export CLARINET_BASE_DIR="${CLARINET_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DTYPE=bfloat16
# AMD RDNA3 arch hint for PyTorch ROCm:
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

mkdir -p "$CLARINET_BASE_DIR"

# -----------------------------------------------------------------------------
# Venv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu --group dev      # bootstraps everything except torch
# Swap in the ROCm torch build (overrides the cpu wheel from uv sync).
# torch 2.9.1's ROCm wheels are only published at rocm6.3 and rocm6.4;
# matching the system ROCm runtime version is best (we assume 6.4 since
# that's what AMD's current WSL installer ships).
.venv/bin/pip install --quiet --force-reinstall torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/rocm6.4
source .venv/bin/activate

# Sanity: GPU must be visible
python -c "import torch; assert torch.cuda.is_available(), 'CUDA/ROCm device not visible'; print('Device:', torch.cuda.get_device_name(0))"

# -----------------------------------------------------------------------------
# wandb (optional, default off — set WANDB_RUN to enable)

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report header

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Data: climbmix (general) + FineMath (reasoning instrument)

# Climbmix: 170 shards covers d24 at ratio=8 with safe margin.
python -m nanochat.dataset -n 8           # 8 shards minimum for tokenizer step
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# FineMath: 50 shards covers the 30% reasoning slice at d24.
python -m clarinet.prepare_reasoning_data -n 50 &
REASONING_PID=$!

# Tokenizer (re)train. Required because clarinet added 3 special tokens, so any
# pre-existing tokenizer artifact would have wrong vocab indices.
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for climbmix download to complete..."
wait $DATASET_DOWNLOAD_PID
echo "Waiting for FineMath prep to complete..."
wait $REASONING_PID

# -----------------------------------------------------------------------------
# Pretraining — the long stage (~12-15 days at d24 on a 7900 XT)
# device-batch-size=2 is the conservative starting point for 20 GB VRAM.
# Bump to 4 if you've confirmed it fits in a smoke test; total-batch-size
# stays the same (just fewer gradient-accumulation microsteps).

python -m scripts.clarinet_train \
    --reasoning-mix-ratio=0.3 --p-uncond=0.1 \
    --depth=24 --target-param-data-ratio=8 \
    --device-batch-size=2 \
    --total-batch-size=524288 \
    --save-every=500 \
    --run=$WANDB_RUN

# Base model eval (~30 min on single GPU)
python -m scripts.base_eval --device-batch-size=2

# -----------------------------------------------------------------------------
# SFT (~4-6 hrs)

curl -L -o "$CLARINET_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft --device-batch-size=2 --run=$WANDB_RUN

# Vanilla chat eval for reference accuracy without IV combination
python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# IV guidance weight sweep — the actual clarinet experiment.
# Trimmed for local time budget: 4 weights (anchors + likely-optimal range),
# 3 tasks (one reasoning-heavy, one categorical, one style/breadth).
# Re-run the full sweep on a rented box later for the publication-grade numbers.

python -m scripts.iv_eval -i sft \
    -a GSM8K,ARC-Easy,MMLU \
    --weights 0,1.0,1.5,2.0,3.0

# -----------------------------------------------------------------------------
# Bonus: chat with a chosen weight after the run finishes
#   python -m scripts.clarinet_cli --iv-weight=2.0 -p "Prove sqrt(2) is irrational."

python -m nanochat.report generate
echo "Done. Report at: $CLARINET_BASE_DIR/report/report.md"
