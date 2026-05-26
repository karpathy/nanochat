bash -lc '
set -euo pipefail

# Paste this into RunPod "Container Start Command".
# It prepares the pod only. Start preflight/speedrun manually after the pod is up.

export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
export NANOCHAT_REPO_URL="${NANOCHAT_REPO_URL:-https://github.com/egcode/nanochat.git}"
export NANOCHAT_BRANCH="${NANOCHAT_BRANCH:-master}"
export NANOCHAT_DIR="${NANOCHAT_DIR:-/workspace/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat-cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_RUN="${WANDB_RUN:-nanochat-speedrun}"
export WANDB_MODE="${WANDB_MODE:-online}"

fail() {
  echo "ERROR: $*" >&2
  echo "Bootstrap failed; exiting instead of keeping an idle paid pod alive." >&2
  exit 1
}

mkdir -p /workspace "$NANOCHAT_BASE_DIR"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git nano ninja-build build-essential python3-dev python3-venv \
    libnuma1 libnuma-dev pkg-config curl ca-certificates
fi

[[ -n "${GIT_USER_NAME:-}" ]] && git config --global user.name "$GIT_USER_NAME"
[[ -n "${GIT_USER_EMAIL:-}" ]] && git config --global user.email "$GIT_USER_EMAIL"

clone_url="$NANOCHAT_REPO_URL"
if [[ -n "${GITHUB_PAT:-}" && "$NANOCHAT_REPO_URL" == https://github.com/* ]]; then
  clone_url="${NANOCHAT_REPO_URL/https:\/\//https:\/\/${GITHUB_PAT}@}"
fi

if [[ -d "$NANOCHAT_DIR/.git" ]]; then
  git -C "$NANOCHAT_DIR" fetch origin "$NANOCHAT_BRANCH"
  git -C "$NANOCHAT_DIR" checkout "$NANOCHAT_BRANCH"
  git -C "$NANOCHAT_DIR" pull --ff-only origin "$NANOCHAT_BRANCH"
else
  [[ -e "$NANOCHAT_DIR" ]] && fail "$NANOCHAT_DIR exists but is not a git repo"
  git clone --branch "$NANOCHAT_BRANCH" "$clone_url" "$NANOCHAT_DIR"
  git -C "$NANOCHAT_DIR" remote set-url origin "$NANOCHAT_REPO_URL"
fi

cd "$NANOCHAT_DIR"
[[ -f pyproject.toml ]] || fail "pyproject.toml not found"
[[ -f runs/preflight_artifacts.sh ]] || fail "runs/preflight_artifacts.sh not found"
[[ -f runs/speedrun.sh ]] || fail "runs/speedrun.sh not found"
[[ -w "$NANOCHAT_BASE_DIR" ]] || fail "NANOCHAT_BASE_DIR is not writable: $NANOCHAT_BASE_DIR"

echo "Bootstrap OK. Next commands:"
echo "  cd $NANOCHAT_DIR"
echo "  bash runs/preflight_artifacts.sh"
echo "  screen -L -Logfile \"\$NANOCHAT_BASE_DIR/speedrun.log\" -S speedrun bash runs/speedrun.sh"

[[ -x /start.sh ]] || fail "/start.sh was not found"
exec /start.sh
'
