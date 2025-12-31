# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# install the repo dependencies
uv sync --extra gpu
# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

