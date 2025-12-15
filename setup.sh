#!/usr/bin/env bash
# Setup nanochat after cloning the repo.
# - initializes the tools submodule (lm-evaluation-harness)
# - creates a uv virtualenv
# - installs deps (choose gpu|cpu extra)
# - builds the Rust tokenizer extension

set -euo pipefail

# -----------------------------
# Helpers
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

extra="${1:-gpu}"
if [[ "$extra" != "gpu" && "$extra" != "cpu" ]]; then
  echo "Usage: bash setup.sh [gpu|cpu]" >&2
  exit 1
fi

echo "[setup] Initializing submodules (tools/lm-eval)..."
git submodule update --init --recursive

echo "[setup] Ensuring uv is installed..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] Ensuring Rust toolchain..."
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck source=/dev/null
command -v cargo >/dev/null 2>&1 || source "$HOME/.cargo/env"

echo "[setup] Creating virtual environment (.venv)..."
[ -d ".venv" ] || uv venv

echo "[setup] Installing Python deps (extra=$extra)..."
uv sync --extra "$extra"

echo "[setup] Building Rust tokenizer (rustbpe)..."
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "[setup] CONDA_PREFIX detected; unsetting to avoid conflicts with VIRTUAL_ENV during build..."
  unset CONDA_PREFIX
fi
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "[setup] Done. Activate with: source .venv/bin/activate"
