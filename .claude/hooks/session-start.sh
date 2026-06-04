#!/bin/bash
# SessionStart hook — install Clarinet's dependencies so tests (and any linters)
# run in Claude Code on the web sessions.
#
# Why not `uv sync --extra cpu`? The project pins torch from the PyTorch CDN
# (download.pytorch.org), which the web environment's network policy blocks
# (HTTP 403). The exact pinned torch==2.9.1 is also published on PyPI, which is
# reachable, so we install the project from PyPI via `uv pip` instead. The PyPI
# linux torch wheel is CUDA-enabled but runs fine on CPU-only nodes.
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Create the project virtualenv (idempotent; matches .python-version = 3.10).
uv venv --python 3.10 .venv

# Install the project (editable) + runtime deps + the `dev` group (pytest, ...).
# torch==2.9.1 and friends resolve from PyPI; `datasets` brings in pyarrow, etc.
VIRTUAL_ENV="$PROJECT_DIR/.venv" uv pip install -e . --group dev

# Persist venv activation so the agent's shell finds python/pytest for the session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export VIRTUAL_ENV=\"$PROJECT_DIR/.venv\""
    echo "export PATH=\"$PROJECT_DIR/.venv/bin:\$PATH\""
  } >> "$CLAUDE_ENV_FILE"
fi
