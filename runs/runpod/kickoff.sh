#!/usr/bin/env bash
# Generic local kickoff for RunPod runs.
# Picks a runner script in this repo (runs/runpod/<RUNNER>.sh) and spins up a pod.
#
# Prereqs:
#   1. ~/.config/team-xsa/runpod.env exports HF_TOKEN, WANDB_API_KEY, RUNPOD_TEMPLATE_ID
#   2. The template referenced by RUNPOD_TEMPLATE_ID has docker-start-cmd:
#        bash,-lc,curl -fsSL "$RUNNER_URL" | bash >> /workspace/runner.log 2>&1
#   3. The runner script for this experiment has been pushed to Team-XSA/nanochat
#
# Usage:
#   source ~/.config/team-xsa/runpod.env
#   bash runs/runpod/kickoff.sh d12         # uses runs/runpod/d12.sh
#   bash runs/runpod/kickoff.sh d24         # uses runs/runpod/d24.sh
#   bash runs/runpod/kickoff.sh xsa_d12     # uses runs/runpod/xsa_d12.sh
#
# Optional env overrides:
#   GPU_ID         default: "NVIDIA H100 80GB HBM3"
#   GPU_COUNT      default: 8
#   CLOUD_TYPE     default: SECURE        (COMMUNITY when capacity available, cheaper)
#   DISK_GB        default: 200
#   NANOCHAT_REPO  default: Team-XSA/nanochat
#   NANOCHAT_REF   default: dev
#   WANDB_RUN      default: <RUNNER>
#   POD_NAME       default: <RUNNER>-<timestamp>

set -euo pipefail

RUNNER="${1:-}"
if [ -z "$RUNNER" ]; then
  echo "Usage: bash runs/runpod/kickoff.sh <runner-name>"
  echo "  e.g. bash runs/runpod/kickoff.sh d12"
  exit 1
fi

: "${HF_TOKEN:?HF_TOKEN not set — source ~/.config/team-xsa/runpod.env}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set — source ~/.config/team-xsa/runpod.env}"
: "${RUNPOD_TEMPLATE_ID:?RUNPOD_TEMPLATE_ID not set — create the template once and add it to ~/.config/team-xsa/runpod.env}"

NANOCHAT_REPO="${NANOCHAT_REPO:-Team-XSA/nanochat}"
NANOCHAT_REF="${NANOCHAT_REF:-dev}"
WANDB_RUN="${WANDB_RUN:-$RUNNER}"
RUNNER_URL="${RUNNER_URL:-https://raw.githubusercontent.com/${NANOCHAT_REPO}/${NANOCHAT_REF}/runs/runpod/${RUNNER}.sh}"

GPU_ID="${GPU_ID:-NVIDIA H100 80GB HBM3}"
GPU_COUNT="${GPU_COUNT:-8}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
DISK_GB="${DISK_GB:-200}"
POD_NAME="${POD_NAME:-${RUNNER}-$(date +%Y%m%d-%H%M)}"

echo "Verifying runner URL is reachable: $RUNNER_URL"
if ! curl -sfI "$RUNNER_URL" >/dev/null; then
  echo "ERROR: runner not reachable at $RUNNER_URL"
  echo "  - Did you push runs/runpod/${RUNNER}.sh to ${NANOCHAT_REPO}@${NANOCHAT_REF}?"
  echo "  - Is the repo public?"
  exit 1
fi

export HF_TOKEN WANDB_API_KEY WANDB_RUN RUNNER_URL NANOCHAT_REPO NANOCHAT_REF
ENV_JSON=$(python3 - <<'PY'
import json, os
keys = ["HF_TOKEN","WANDB_API_KEY","WANDB_RUN","RUNNER_URL","NANOCHAT_REPO","NANOCHAT_REF"]
print(json.dumps({k: os.environ[k] for k in keys if k in os.environ}))
PY
)

echo "Creating pod:"
echo "  name        = $POD_NAME"
echo "  template    = $RUNPOD_TEMPLATE_ID"
echo "  runner      = $RUNNER_URL"
echo "  gpu         = $GPU_COUNT × $GPU_ID"
echo "  cloud       = $CLOUD_TYPE"
echo "  disk        = ${DISK_GB} GB"

runpodctl pod create \
  --name "$POD_NAME" \
  --template-id "$RUNPOD_TEMPLATE_ID" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --cloud-type "$CLOUD_TYPE" \
  --container-disk-in-gb "$DISK_GB" \
  --env "$ENV_JSON"

echo
echo "Logs (after pod boots):"
echo "  POD_ID=\$(runpodctl pod list --name '$POD_NAME' -o json | jq -r '.[0].id')"
echo "  runpodctl ssh info \$POD_ID"
echo "  ssh <user>@<host> 'tail -f /workspace/runner.log'"
echo
echo "Wandb: project=nanochat / nanochat-sft, run name: $WANDB_RUN"
echo "HF (success):  https://huggingface.co/haydenfree/nanochat-d12-baseline"
echo "HF (failure):  https://huggingface.co/haydenfree/nanochat-d12-baseline/tree/main/_failures"
