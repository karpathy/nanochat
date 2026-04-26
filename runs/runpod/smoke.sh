#!/usr/bin/env bash
# Minimal smoke test. Runs INSIDE a RunPod pod.
# Validates: pod boot, env-var injection, git clone, uv sync, GPU torch,
# tokenizer + base_train code paths, HF upload, runpodctl self-delete.
# Does NOT test: multi-GPU, FP8, full training horizon, SFT, eval.
#
# Sized for a 1-GPU pod, completes in ~3-4 min wall clock.
# Kick off with: GPU_COUNT=1 bash runs/runpod/kickoff.sh smoke
#
# Required env: HF_TOKEN, WANDB_API_KEY
# Auto-set by RunPod: RUNPOD_POD_ID, RUNPOD_API_KEY

set -euo pipefail

NANOCHAT_REPO="${NANOCHAT_REPO:-Team-XSA/nanochat}"
NANOCHAT_REF="${NANOCHAT_REF:-dev}"
HF_REPO="${HF_REPO:-haydenfree/nanochat-d12-baseline}"
WANDB_RUN="${WANDB_RUN:-smoke}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
HF_PATH_PREFIX="_smoke/${TS}"

WORKDIR="/workspace/nanochat"
LOG_FILE="/workspace/runner.log"
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

mkdir -p /workspace
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[smoke] $(date -Iseconds) starting on pod=$RUNPOD_POD_ID"

cleanup() {
  local rc=$?
  set +e
  echo "[smoke] cleanup: exit code $rc"

  # Always upload the runner log (success or failure) so we can see what happened
  mkdir -p /tmp/smoke-out
  cp /workspace/*.log /tmp/smoke-out/ 2>/dev/null || true
  echo "rc=$rc ts=$TS pod=$RUNPOD_POD_ID" > /tmp/smoke-out/result.txt
  [ -d "$WORKDIR" ] && (cd "$WORKDIR" && git rev-parse HEAD 2>/dev/null > /tmp/smoke-out/git-head.txt || true)
  huggingface-cli upload "$HF_REPO" /tmp/smoke-out "$HF_PATH_PREFIX" \
    --repo-type model --commit-message "smoke rc=$rc $TS" || \
    echo "[smoke] WARN: HF upload failed"

  echo "[smoke] artifacts: https://huggingface.co/$HF_REPO/tree/main/$HF_PATH_PREFIX"
  echo "[smoke] self-deleting pod $RUNPOD_POD_ID"
  runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 || \
    curl -sS -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
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
git checkout "$NANOCHAT_REF"
echo "[smoke] HEAD = $(git rev-parse HEAD)"

# Env + uv
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
pip install --quiet --upgrade huggingface_hub

# GPU sanity
python -c "import torch; print('[smoke] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# Minimum dataset + tokenizer (1 shard, 50M chars — enough for the tokenizer
# to train on AND for base_train to consume 20 iterations of tokens)
python -m nanochat.dataset -n 1
python -m scripts.tok_train --max-chars=50000000

# Tiny base_train. Params from base_train.py docstring (the CPU smoke), adjusted
# slightly for GPU. depth=4, 20 iterations. Should finish in ~30s.
NPROC=$(nvidia-smi -L | wc -l)
echo "[smoke] training on $NPROC GPU(s)"
torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=512 \
    --num-iterations=20 \
    --eval-every=10 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --run="$WANDB_RUN"

echo "[smoke] $(date -Iseconds) base_train complete — smoke passed"
# trap cleanup handles HF upload + self-delete
