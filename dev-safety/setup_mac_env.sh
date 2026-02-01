#!/bin/bash
set -euo pipefail

# Setup script for running nanochat on macOS (Apple Silicon or Intel)
# This sets up the environment to run the d12 checkpoint on CPU or MPS (Metal Performance Shaders)

echo "=========================================="
echo "nanochat macOS Environment Setup"
echo "=========================================="

# Check if we're in the nanochat directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the nanochat directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv installation failed or not in PATH"
    echo "Please add ~/.local/bin to your PATH and try again"
    exit 1
fi

echo ""
echo "Creating virtual environment with CPU/MPS support..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
    echo "Created .venv"
fi

# Install dependencies with CPU extras (includes MPS support on macOS)
echo "Installing dependencies (this may take a few minutes)..."
uv sync --extra cpu

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the d12 checkpoint:"
echo "  cd dev-safety && ./run_d12_mac.sh"
echo ""
