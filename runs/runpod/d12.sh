#!/usr/bin/env bash
# d12 baseline runner. Runs INSIDE a RunPod pod.
# Pipeline: tokenizer -> base_train -> base_eval -> SFT -> chat_eval -> report.
# On exit:
#   success  -> upload final cache to HF, self-delete pod
#   failure  -> upload logs + report dir to HF under _failures/, self-delete pod
#               (set UPLOAD_FAILURE_CACHE=1 to also dump partial cache for offline debug)
#
# Required env (passed via runpodctl --env at pod-create):
#   HF_TOKEN, WANDB_API_KEY
# Optional env:
#   WANDB_RUN              default: d12
#   NANOCHAT_REPO          default: Team-XSA/nanochat
#   NANOCHAT_REF           default: dev
#   HF_REPO                default: haydenfree/nanochat-d12-baseline
#   BACKUP_INTERVAL        default: 300 (seconds between background HF uploads)
#   UPLOAD_FAILURE_CACHE   default: 0
# Auto-set by RunPod:
#   RUNPOD_POD_ID, RUNPOD_API_KEY (pod-scoped)

set -euo pipefail

NANOCHAT_REPO="${NANOCHAT_REPO:-Team-XSA/nanochat}"
NANOCHAT_REF="${NANOCHAT_REF:-dev}"
HF_REPO="${HF_REPO:-haydenfree/nanochat-d12-baseline}"
WANDB_RUN="${WANDB_RUN:-d12}"
BACKUP_INTERVAL="${BACKUP_INTERVAL:-300}"
UPLOAD_FAILURE_CACHE="${UPLOAD_FAILURE_CACHE:-0}"

WORKDIR="/workspace/nanochat"
LOG_FILE="/workspace/runner.log"
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
BACKUP_PID=""

mkdir -p /workspace
# NOTE: dockerStartCmd already redirects stdout/stderr to $LOG_FILE.
# Don't add a second tee here — would write every line twice.

echo "[runner] $(date -Iseconds) starting on pod=$RUNPOD_POD_ID"
echo "[runner] repo=$NANOCHAT_REPO ref=$NANOCHAT_REF hf_repo=$HF_REPO wandb_run=$WANDB_RUN"

# Bootstrap huggingface_hub system-wide so the cleanup trap can upload logs
# even if we fail before the venv is activated.
{ pip3 install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  python3 -m pip install --break-system-packages --quiet --upgrade huggingface_hub 2>&1 || \
  echo "[runner] WARN: could not pre-install huggingface_hub; cleanup uploads may fail"; } || true

cleanup() {
  local rc=$?
  set +e
  echo "[runner] cleanup: exit code $rc at $(date -Iseconds)"
  if [ -n "$BACKUP_PID" ] && kill -0 "$BACKUP_PID" 2>/dev/null; then
    kill "$BACKUP_PID" 2>/dev/null || true
  fi

  local TS
  TS=$(date -u +%Y%m%dT%H%M%SZ)

  if [ "$rc" -eq 0 ]; then
    echo "[runner] success — final upload to $HF_REPO"
    if [ -d "$NANOCHAT_BASE_DIR" ]; then
      # Skip the climbmix dataset shards (~2GB of public data, not model artifacts)
      hf upload "$HF_REPO" "$NANOCHAT_BASE_DIR" . \
        --repo-type model --commit-message "final rc=0 $TS" \
        --exclude "base_data_climbmix/**" --exclude "wandb/**" || \
        echo "[runner] WARN: final upload failed"
    fi
    # Also upload the runner log so we have a permanent record of this successful run.
    if [ -f "$LOG_FILE" ]; then
      hf upload "$HF_REPO" "$LOG_FILE" "_runs/${TS}/runner.log" \
        --repo-type model --commit-message "runner log $TS" || \
        echo "[runner] WARN: runner.log upload failed"
    fi
  else
    echo "[runner] failure rc=$rc — dumping logs to HF for offline debug"
    mkdir -p /tmp/failure
    cp /workspace/*.log /tmp/failure/ 2>/dev/null || true
    [ -d "$NANOCHAT_BASE_DIR/report" ] && cp -r "$NANOCHAT_BASE_DIR/report" /tmp/failure/ 2>/dev/null || true
    [ -d "$WORKDIR" ] && (cd "$WORKDIR" && git rev-parse HEAD 2>/dev/null > /tmp/failure/git-head.txt || true)

    hf upload "$HF_REPO" /tmp/failure "_failures/${TS}-rc${rc}/logs" \
      --repo-type model --commit-message "failure rc=$rc logs $TS" || \
      echo "[runner] WARN: log upload failed"

    if [ "$UPLOAD_FAILURE_CACHE" = "1" ] && [ -d "$NANOCHAT_BASE_DIR" ]; then
      echo "[runner] UPLOAD_FAILURE_CACHE=1 — also dumping partial cache (may be slow)"
      hf upload "$HF_REPO" "$NANOCHAT_BASE_DIR" "_failures/${TS}-rc${rc}/cache" \
        --repo-type model --commit-message "failure rc=$rc cache $TS" \
        --exclude "base_data_climbmix/**" --exclude "wandb/**" || true
    fi
    echo "[runner] failure artifacts: https://huggingface.co/$HF_REPO/tree/main/_failures/${TS}-rc${rc}"
  fi

  echo "[runner] self-deleting pod $RUNPOD_POD_ID"
  # REST API first — pod-scoped key has delete permission and the API is reliable.
  # The pod's preinstalled runpodctl is unreliable (often missing config or 'pod' subcommand).
  if curl -fsS -X DELETE \
       -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
       "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" 2>&1; then
    echo "[runner] REST delete request accepted"
  else
    echo "[runner] REST delete failed, trying runpodctl as fallback"
    runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 || \
    runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 || \
    echo "[runner] WARN: all delete methods failed — pod may need manual cleanup"
  fi
  exit "$rc"
}
trap cleanup EXIT

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID must be set (auto by RunPod)}"

rm -rf "$WORKDIR"
git clone "https://github.com/${NANOCHAT_REPO}.git" "$WORKDIR"
cd "$WORKDIR"
# `--` disambiguates ref-vs-file (some images create a `dev` file in HOME)
git checkout "$NANOCHAT_REF" --
echo "[runner] HEAD = $(git rev-parse HEAD)"

sed -i 's/--depth=24/--depth=12/' runs/speedrun.sh
sed -i 's/ --target-param-data-ratio=8//' runs/speedrun.sh
# Inject `set -euo pipefail` so a mid-pipeline failure (e.g. chat_sft) propagates
# as rc!=0 instead of being silently swallowed by the next command.
sed -i '1a set -euo pipefail' runs/speedrun.sh
echo "[runner] speedrun.sh edits applied:"
grep -n 'depth\|target-param\|set -e' runs/speedrun.sh || true

# Explicit venv setup BEFORE speedrun.sh so we can run diagnostic probes
# inside the venv. speedrun.sh's uv sync is idempotent (no-op the second time).
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
pip install --quiet --upgrade huggingface_hub

# Ensure HF token flows to the kernels lib (some libs read HF_HUB_TOKEN, not HF_TOKEN)
export HF_HUB_TOKEN="${HF_TOKEN}"

# Bump kernels to latest — pyproject pins >=0.11.7 and uv often picks exactly that;
# 0.11.x had kernel-resolution bugs that affect FA3 loading silently.
echo "[runner] upgrading kernels lib for FA3 reliability"
uv pip install --quiet --upgrade 'kernels>=0.13.0' 2>&1 || \
  echo "[runner] WARN: kernels upgrade failed (continuing)"

# Install hf_transfer — runpod base image sets HF_HUB_ENABLE_HF_TRANSFER=1, which
# makes huggingface_hub raise ValueError if the package is missing. chat_sft loads
# HuggingFaceTB/smol-smoltalk via datasets and crashes without this.
echo "[runner] installing hf_transfer for SFT dataset download"
uv pip install --quiet hf_transfer 2>&1 || echo "[runner] WARN: hf_transfer install failed"

# FA3 diagnostic probe — surfaces real errors (nanochat silently swallows them).
# Non-fatal: SDPA fallback is automatic. We want this output in the log
# regardless of outcome so we can decide what to do about FA3.
echo "[runner] === FA3 PROBE BEGIN ==="
python "$WORKDIR/runs/runpod/probe_fa3.py" || echo "[runner] FA3 probe reported issues (non-fatal — continuing with SDPA fallback)"
echo "[runner] === FA3 PROBE END ==="

(
  while true; do
    sleep "$BACKUP_INTERVAL"
    if [ -d "$NANOCHAT_BASE_DIR" ]; then
      hf upload "$HF_REPO" "$NANOCHAT_BASE_DIR" . \
        --repo-type model \
        --commit-message "checkpoint $(date -Iseconds)" \
        --exclude "base_data_climbmix/**" --exclude "wandb/**" \
        >> /workspace/backup.log 2>&1 || true
    fi
  done
) &
BACKUP_PID=$!
echo "[runner] backup loop pid=$BACKUP_PID interval=${BACKUP_INTERVAL}s"

export WANDB_RUN
WANDB_RUN="$WANDB_RUN" bash runs/speedrun.sh

# Verify expected pipeline outputs — speedrun.sh historically didn't `set -e`;
# we patched it above, but double-check the artifacts that matter for the d12 baseline.
echo "[runner] verifying pipeline outputs"
missing=()
for required in base_checkpoints/d12 chatsft_checkpoints/d12 tokenizer report; do
  if [ ! -d "$NANOCHAT_BASE_DIR/$required" ]; then
    missing+=("$required")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "[runner] FAIL: pipeline finished but missing expected artifacts: ${missing[*]}"
  exit 1
fi
echo "[runner] all expected artifacts present"

echo "[runner] $(date -Iseconds) pipeline complete"
