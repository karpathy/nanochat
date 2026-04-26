#!/usr/bin/env bash
# d12 SFT-only resume runner. Runs INSIDE a RunPod pod.
#
# Use case: the d12 base_train + base_eval already succeeded and uploaded to HF,
# but chat_sft failed (e.g., missing hf_transfer package). Instead of re-running
# the whole pipeline, this runner:
#   1. Downloads base_checkpoints/d12/ + tokenizer/ from HF
#   2. Installs hf_transfer (the actual SFT bug fix)
#   3. Runs chat_sft + chat_eval directly (skips speedrun.sh)
#   4. Uploads chatsft_checkpoints/ + chat_eval results + report to HF
#   5. Self-deletes
#
# Required env: HF_TOKEN, WANDB_API_KEY
# Optional env:
#   WANDB_RUN     default: d12-sft
#   NANOCHAT_REPO default: Team-XSA/nanochat
#   NANOCHAT_REF  default: dev
#   HF_REPO       default: haydenfree/nanochat-d12-baseline (where the base lives)

set -euo pipefail

NANOCHAT_REPO="${NANOCHAT_REPO:-Team-XSA/nanochat}"
NANOCHAT_REF="${NANOCHAT_REF:-dev}"
HF_REPO="${HF_REPO:-haydenfree/nanochat-d12-baseline}"
WANDB_RUN="${WANDB_RUN:-d12-sft}"

WORKDIR="/workspace/nanochat"
LOG_FILE="/workspace/runner.log"
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

mkdir -p /workspace
echo "[sft] $(date -Iseconds) starting on pod=$RUNPOD_POD_ID"
echo "[sft] resuming from base checkpoint at $HF_REPO"

# Bootstrap huggingface_hub system-wide so cleanup can upload logs even on early failure.
{ pip3 install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  python3 -m pip install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  echo "[sft] WARN: could not pre-install huggingface_hub"; } || true

cleanup() {
  local rc=$?
  set +e
  echo "[sft] cleanup: exit code $rc at $(date -Iseconds)"

  local TS
  TS=$(date -u +%Y%m%dT%H%M%SZ)

  if [ "$rc" -eq 0 ]; then
    echo "[sft] success — uploading chatsft_checkpoints + report + log"
    # Only upload the SFT-specific subdirs so we don't re-upload base.
    for subdir in chatsft_checkpoints report; do
      if [ -d "$NANOCHAT_BASE_DIR/$subdir" ]; then
        hf upload "$HF_REPO" "$NANOCHAT_BASE_DIR/$subdir" "$subdir" \
          --repo-type model --commit-message "$subdir SFT-resume rc=0 $TS" || \
          echo "[sft] WARN: $subdir upload failed"
      fi
    done
    if [ -f "$LOG_FILE" ]; then
      hf upload "$HF_REPO" "$LOG_FILE" "_runs/${TS}-sft/runner.log" \
        --repo-type model --commit-message "SFT runner log $TS" || \
        echo "[sft] WARN: runner.log upload failed"
    fi
  else
    echo "[sft] failure rc=$rc — dumping logs"
    mkdir -p /tmp/failure
    cp /workspace/*.log /tmp/failure/ 2>/dev/null || true
    [ -d "$NANOCHAT_BASE_DIR/report" ] && cp -r "$NANOCHAT_BASE_DIR/report" /tmp/failure/ 2>/dev/null || true
    [ -d "$WORKDIR" ] && (cd "$WORKDIR" && git rev-parse HEAD 2>/dev/null > /tmp/failure/git-head.txt || true)
    hf upload "$HF_REPO" /tmp/failure "_failures/${TS}-sft-rc${rc}/logs" \
      --repo-type model --commit-message "SFT-resume failure rc=$rc $TS" || \
      echo "[sft] WARN: log upload failed"
  fi

  echo "[sft] self-deleting pod $RUNPOD_POD_ID"
  if curl -fsS -X DELETE \
       -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
       "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" 2>&1; then
    echo "[sft] REST delete request accepted"
  else
    echo "[sft] REST delete failed, trying runpodctl as fallback"
    runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 || \
    runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 || \
    echo "[sft] WARN: all delete methods failed — pod may need manual cleanup"
  fi
  exit "$rc"
}
trap cleanup EXIT

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set (auto by RunPod)}"

# Clone fork
rm -rf "$WORKDIR"
git clone "https://github.com/${NANOCHAT_REPO}.git" "$WORKDIR"
cd "$WORKDIR"
git checkout "$NANOCHAT_REF" --
echo "[sft] HEAD = $(git rev-parse HEAD)"

# Env + uv
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
pip install --quiet --upgrade huggingface_hub
export HF_HUB_TOKEN="${HF_TOKEN}"

# Install hf_transfer — THE actual fix for the previous SFT failure.
echo "[sft] installing hf_transfer (the bug from last run)"
uv pip install --quiet hf_transfer

# Pull tokenizer + base checkpoint from HF — skip base_train entirely
echo "[sft] downloading tokenizer and base_checkpoints/d12 from $HF_REPO"
hf download "$HF_REPO" \
  --include "tokenizer/**" \
  --include "base_checkpoints/d12/**" \
  --local-dir "$NANOCHAT_BASE_DIR" \
  --repo-type model

ls -la "$NANOCHAT_BASE_DIR/base_checkpoints/d12/" || true
ls -la "$NANOCHAT_BASE_DIR/tokenizer/" || true

# Also need identity_conversations.jsonl for SFT (speedrun.sh normally fetches it)
echo "[sft] fetching identity_conversations.jsonl"
curl -L -fsS -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run only SFT + chat_eval + report. NOT speedrun.sh (which would re-do base_train).
NPROC=$(nvidia-smi -L | wc -l)
echo "[sft] running chat_sft on $NPROC GPUs"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- \
  --device-batch-size=16 --run="$WANDB_RUN"

echo "[sft] running chat_eval"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_eval -- -i sft

echo "[sft] regenerating report (will include new SFT sections)"
python -m nanochat.report generate || true

# Verify SFT artifacts exist before declaring success
if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ]; then
  echo "[sft] FAIL: chatsft_checkpoints/ missing after chat_sft"
  exit 1
fi

echo "[sft] $(date -Iseconds) SFT pipeline complete"
