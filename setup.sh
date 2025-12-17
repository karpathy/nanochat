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
# -----------------------------
# Rust build output: avoid building into repo (may be on NFS/overlay and cause mmap 0-len errors)
export PATH="$HOME/.cargo/bin:$PATH"
# shellcheck source=/dev/null
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# Prefer node-local scratch for Cargo artifacts
if [ -d /tmp ] && [ -w /tmp ]; then
  export CARGO_TARGET_DIR="/tmp/cargo-target-${USER:-root}/nanochat"
else
  export CARGO_TARGET_DIR="$repo_root/.cargo-target"
fi
mkdir -p "$CARGO_TARGET_DIR"

echo "[setup] Using CARGO_TARGET_DIR=$CARGO_TARGET_DIR"

extra="${1:-gpu}"
if [[ "$extra" != "gpu" && "$extra" != "cpu" ]]; then
  echo "Usage: bash setup.sh [gpu|cpu]" >&2
  exit 1
fi

echo "[setup] Initializing submodules (tools/lm-eval)..."
echo "[setup] This would take some time to download all the benchmarks in lm-eval"
git submodule update --init --recursive

echo "[setup] Ensuring uv is installed..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] Ensuring Rust toolchain..."

# Always ensure cargo bin is on PATH (important for non-interactive shells / root)
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v rustup >/dev/null 2>&1; then
  echo "[setup] rustup not found; installing..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# Load cargo env if present (ignore if missing)
# shellcheck source=/dev/null
if [ -f "$HOME/.cargo/env" ]; then
  source "$HOME/.cargo/env"
fi

# After installation/sourcing, ensure cargo exists
if ! command -v cargo >/dev/null 2>&1; then
  echo "[setup] ERROR: cargo not found even after rustup install. PATH=$PATH" >&2
  exit 1
fi

# Ensure a default toolchain is configured, otherwise maturin/cargo metadata will fail.
# This fixes: "rustup could not choose a version of cargo to run ... no default is configured."
if rustup show active-toolchain >/dev/null 2>&1; then
  echo "[setup] Active Rust toolchain: $(rustup show active-toolchain | head -n 1)"
else
  echo "[setup] No active Rust toolchain; setting default to stable..."
  rustup default stable
fi

# (Optional but nice) make sure stable is installed even if default points elsewhere
rustup toolchain install stable >/dev/null 2>&1 || true

echo "[setup] Rust: $(rustc --version 2>/dev/null || echo 'rustc not found')"
echo "[setup] Cargo: $(cargo --version 2>/dev/null || echo 'cargo not found')"


echo "[setup] Creating virtual environment (.venv)..."
[ -d ".venv" ] || uv venv

echo "[setup] Cleaning rustbpe build artifacts (safe)..."
rm -rf rustbpe/target || true

echo "[setup] Installing Python deps (extra=$extra)..."
echo "[setup] This will iterate all benchmarks, may take a long time"
UV_LOG_LEVEL=debug uv sync --extra "$extra" -v

echo "[setup] Building Rust tokenizer (rustbpe)..."
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "[setup] CONDA_PREFIX detected; unsetting to avoid conflicts with VIRTUAL_ENV during build..."
  unset CONDA_PREFIX
fi
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "[setup] Done. Activate with: source .venv/bin/activate"
