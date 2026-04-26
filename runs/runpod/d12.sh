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
      huggingface-cli upload "$HF_REPO" "$NANOCHAT_BASE_DIR" . \
        --repo-type model --commit-message "final rc=0 $TS" || \
        echo "[runner] WARN: final upload failed"
    fi
  else
    echo "[runner] failure rc=$rc — dumping logs to HF for offline debug"
    mkdir -p /tmp/failure
    cp /workspace/*.log /tmp/failure/ 2>/dev/null || true
    [ -d "$NANOCHAT_BASE_DIR/report" ] && cp -r "$NANOCHAT_BASE_DIR/report" /tmp/failure/ 2>/dev/null || true
    [ -d "$WORKDIR" ] && (cd "$WORKDIR" && git rev-parse HEAD 2>/dev/null > /tmp/failure/git-head.txt || true)

    huggingface-cli upload "$HF_REPO" /tmp/failure "_failures/${TS}-rc${rc}/logs" \
      --repo-type model --commit-message "failure rc=$rc logs $TS" || \
      echo "[runner] WARN: log upload failed"

    if [ "$UPLOAD_FAILURE_CACHE" = "1" ] && [ -d "$NANOCHAT_BASE_DIR" ]; then
      echo "[runner] UPLOAD_FAILURE_CACHE=1 — also dumping partial cache (may be slow)"
      huggingface-cli upload "$HF_REPO" "$NANOCHAT_BASE_DIR" "_failures/${TS}-rc${rc}/cache" \
        --repo-type model --commit-message "failure rc=$rc cache $TS" || true
    fi
    echo "[runner] failure artifacts: https://huggingface.co/$HF_REPO/tree/main/_failures/${TS}-rc${rc}"
  fi

  echo "[runner] self-deleting pod $RUNPOD_POD_ID"
  # Preinstalled runpodctl may be older (legacy 'remove pod') or newer ('pod delete').
  # Try new, then legacy, then REST API. -fsS makes curl fail loudly on HTTP errors.
  if runpodctl pod delete "$RUNPOD_POD_ID" 2>&1; then
    :
  elif runpodctl remove pod "$RUNPOD_POD_ID" 2>&1; then
    :
  else
    echo "[runner] runpodctl delete failed via both syntaxes, using REST API"
    curl -fsS -X DELETE \
      -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" 2>&1 || \
      echo "[runner] WARN: REST delete also failed — pod may need manual cleanup"
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
echo "[runner] speedrun.sh edits applied:"
grep -n 'depth\|target-param' runs/speedrun.sh || true

pip install --quiet --upgrade huggingface_hub

(
  while true; do
    sleep "$BACKUP_INTERVAL"
    if [ -d "$NANOCHAT_BASE_DIR" ]; then
      huggingface-cli upload "$HF_REPO" "$NANOCHAT_BASE_DIR" . \
        --repo-type model \
        --commit-message "checkpoint $(date -Iseconds)" >> /workspace/backup.log 2>&1 || true
    fi
  done
) &
BACKUP_PID=$!
echo "[runner] backup loop pid=$BACKUP_PID interval=${BACKUP_INTERVAL}s"

export WANDB_RUN
WANDB_RUN="$WANDB_RUN" bash runs/speedrun.sh

echo "[runner] $(date -Iseconds) pipeline complete"
