#!/bin/bash

# Nanobot: same pipeline as speedrun — Pretrain (base) → SFT (chat) → serve.
# Auto-detects your GPU configuration (count, VRAM, SM version) and selects
# appropriate depth, batch size, and precision so the run works on consumer
# PC graphics cards (RTX 30/40/50 series) as well as data-center nodes.
# Checkpoints go under model-tag "nanobot" (base_checkpoints/nanobot, chatsft_checkpoints/nanobot).

# Example: bash runs/nanobot.sh
# With wandb: WANDB_RUN=nanobot bash runs/nanobot.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=nanobot
fi

# ---------------------------------------------------------------------------
# Auto-detect GPU configuration
# ---------------------------------------------------------------------------

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA GPUs detected. Use runs/runcpu.sh for CPU/MPS training."
    exit 1
fi

VRAM_GB=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))")
SM_VER=$(python -c "import torch; cap = torch.cuda.get_device_capability(); print(cap[0] * 10 + cap[1])")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")

echo "=== GPU Configuration ==="
echo "  GPU:   $GPU_NAME"
echo "  Count: $NUM_GPUS"
echo "  VRAM:  ${VRAM_GB}GB per GPU"
echo "  SM:    $SM_VER"

# FP8 training requires SM >= 89 (Ada Lovelace: RTX 40 series and higher, plus H100+).
# RTX 30 series (SM 86) and older use bf16 only.
FP8_FLAG=""
if [ "$SM_VER" -ge 89 ]; then
    FP8_FLAG="--fp8"
    echo "  FP8:   enabled (SM $SM_VER >= 89)"
else
    echo "  FP8:   disabled (SM $SM_VER < 89, training in bf16)"
fi

# Device batch size: reduce when VRAM is limited to avoid OOM.
# The training script compensates automatically via gradient accumulation.
if [ "$VRAM_GB" -ge 40 ]; then
    DEVICE_BATCH_SIZE=16   # 40GB+ (H100, A100, etc.)
elif [ "$VRAM_GB" -ge 20 ]; then
    DEVICE_BATCH_SIZE=8    # 20-40GB (RTX 3090 24GB, RTX 4090 24GB, A10 24GB, etc.)
elif [ "$VRAM_GB" -ge 12 ]; then
    DEVICE_BATCH_SIZE=4    # 12-20GB (RTX 3080 12GB, RTX 4070 12GB, etc.)
elif [ "$VRAM_GB" -ge 8 ]; then
    DEVICE_BATCH_SIZE=2    # 8-12GB (RTX 3070 8GB, RTX 4060 8GB, etc.)
else
    DEVICE_BATCH_SIZE=1    # <8GB
fi
echo "  Batch: $DEVICE_BATCH_SIZE (device_batch_size)"

# Depth: scale model size with total available VRAM across all GPUs.
TOTAL_VRAM=$((VRAM_GB * NUM_GPUS))
if [ "$TOTAL_VRAM" -ge 320 ]; then
    DEPTH=24   # 8x40GB or 4x80GB (full data-center node)
elif [ "$TOTAL_VRAM" -ge 160 ]; then
    DEPTH=20   # 8x20GB or 4x40GB
elif [ "$TOTAL_VRAM" -ge 80 ]; then
    DEPTH=16   # 4x20GB, 2x40GB, or single 80GB
elif [ "$TOTAL_VRAM" -ge 40 ]; then
    DEPTH=12   # 2x20GB or single 40GB
else
    DEPTH=8    # Single consumer GPU (<40GB total)
fi
echo "  Depth: $DEPTH"
echo "=========================="

# Multi-GPU: use torchrun. Single GPU: use python directly.
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH="torchrun --standalone --nproc_per_node=$NUM_GPUS -m"
else
    LAUNCH="python -m"
fi

python -m nanochat.report reset

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# ---------------------------------------------------------------------------
# Base model (pretraining)
# ---------------------------------------------------------------------------
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

$LAUNCH scripts.base_train -- \
  --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE $FP8_FLAG \
  --run="$WANDB_RUN" --model-tag=nanobot
$LAUNCH scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE

# ---------------------------------------------------------------------------
# SFT (chat)
# ---------------------------------------------------------------------------
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

$LAUNCH scripts.chat_sft -- \
  --device-batch-size=$DEVICE_BATCH_SIZE --run="$WANDB_RUN" --model-tag=nanobot
$LAUNCH scripts.chat_eval -- -i sft --model-tag=nanobot

# Serve: python -m scripts.chat_web  (use nanobot checkpoint via env or default latest)

python -m nanochat.report generate
