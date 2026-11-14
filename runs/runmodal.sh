#!/bin/bash

# Modal nanochat runner: wraps speedrun.sh on 8×H100 with idempotent Volume commits.
# Prerequisites: Modal account, `modal setup`, 8×H100 entitlement, local `modal` CLI.
# One command: bash runs/runmodal.sh
# Cost: ~$40–60 (~$3/GPU/hr × 8 × ~1.5–2h); first run higher for image bake + compile.
# Time: ~1.5–2h training + prefetch + image build. Keep laptop awake; run in screen/tmux.

set -euo pipefail

# Fail-fast: require modal on PATH
if ! command -v modal &>/dev/null; then
    echo "ERROR: 'modal' not found on PATH. Install with: pip install modal (or uv tool run modal)" >&2
    echo "Then run 'modal setup' and verify 8×H100 entitlement." >&2
    exit 1
fi

# Print header with cost and time estimate
cat <<'EOF'
╔═════════════════════════════════════════════════════════════════════════════╗
║ nanochat on Modal — 8×H100 GPT-2-grade LLM training                         ║
║                                                                             ║
║ Cost: ~$40–60 (~$3/GPU/hr × 8 GPUs × ~1.5–2h)                              ║
║       First run: higher (image bake + compile)                             ║
║                                                                             ║
║ Time: ~1.5–2h training + prefetch CPU + image build                        ║
║       Keep laptop awake; use screen/tmux (see RUNNING.md)                  ║
║                                                                             ║
║ On completion: tokenizer + SFT checkpoints downloaded to                   ║
║ ~/.cache/nanochat-modal; chat with:                                        ║
║   NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli    ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
EOF

# Spend confirmation (unless RUNMODAL_YES=1)
if [ -z "${RUNMODAL_YES:-}" ]; then
    echo ""
    read -p "Proceed? [y/N] " -r proceed
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Launch Modal app
echo ""
echo "Starting Modal run (runs/modal_run.py)..."
echo ""
modal run runs/modal_run.py
