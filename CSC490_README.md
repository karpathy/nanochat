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
uv run modal setup
uv run modal secret create nanochat-secrets \
    WANDB_API_KEY=<your_wandb_key> \
    HF_TOKEN=hf_<your_hf_token>

# running models - part 2 ablation studies 
- first time: uv run modal run nanochat_modal.py::main
- re-run one ablation after data/tokenizer are already on the volume
uv run modal run nanochat_modal.py::run_baseline
uv run modal run nanochat_modal.py::run_swiglu
uv run modal run nanochat_modal.py::run_rope500k