# env set up 
uv sync

(if you don't have uv:

    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    # install the repo dependencies
    uv sync
)

source .venv/bin/activate

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

(if you have conda activated: 
    unset CONDA_PREFIX && uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
)

# train tokenizer 
python -m nanochat.dataset -n 240

# modal set up 
pip install modal