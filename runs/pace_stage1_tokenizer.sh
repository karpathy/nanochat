#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J nanochat-stage1-tokenizer
#SBATCH -o runs/logs/stage1_%j.out
#SBATCH -e runs/logs/stage1_%j.err

# Stage 1

set -e
cd "$HOME/scratch/nanochat"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
mkdir -p runs/logs

echo "=== Stage 1: Tokenizer ==="
echo "Base dir: $NANOCHAT_BASE_DIR"
echo "Started: $(date)"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
python -m nanochat.report reset
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for full dataset download..."
wait $DATASET_DOWNLOAD_PID

echo "=== Stage 1 complete: $(date) ==="
echo "Dataset and tokenizer ready in $NANOCHAT_BASE_DIR"
