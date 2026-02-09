# https://forums.developer.nvidia.com/t/anyone-got-nanochat-training-working-on-the-dgx-spark/348537/8


# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

python -m nanochat.dataset -n 240

python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
mv eval_bundle "$HOME/.cache/nanochat"


# install cuda 13.1
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=24.04&target_type=deb_local

# torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=20
# step 00000/04357 (0.00%) | loss: 10.397409 | lrm: 1.00 | dt: 392094.88ms | tok/sec: 2,674 | mfu: 0.00 | epoch: 1 | total time: 0.00m

#