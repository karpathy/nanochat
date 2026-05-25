# WATGPU handoff for d36 4B-class nanochat

This repo is now set up for an end-to-end d36 ClimbMix run through:

```bash
bash runs/train_4b.sh
```

The script trains a d36 model, about 3.8B parameters, on ClimbMix, then runs base eval, SFT, chat eval, and report generation. It also downloads the needed corpora and eval data into `NANOCHAT_BASE_DIR`.

## Greedy GPU target

Prefer H200 NVL nodes. The current public WATGPU layout shows:

- Best: `watgpu-500`, 7x NVIDIA H200 NVL, 140.4 GB each.
- Next: `watgpu-800`, H200 indices `0,1,2,4,5` if those exact GPUs are available.
- Next: `watgpu-700`, 4x NVIDIA H200 NVL.

Avoid the 45-48 GB Ada/L40S/A6000 nodes for this default d36 run. Also prefer H200 over Blackwell for this repo right now: the local Flash Attention 3 path is Hopper-only, while Blackwell falls back to PyTorch SDPA.

## Batch submission shape

Use batch mode from the login node. Do not run training directly on the login node.

Example Slurm wrapper:

```bash
#!/bin/bash
#SBATCH --partition=ALL
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH -o nanochat4b-%j.out
#SBATCH -e nanochat4b-%j.err

set -euo pipefail
cd /path/to/nanochat

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-4b}"
export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix}"
export MODEL_DEPTH=36
export TARGET_PARAM_DATA_RATIO=12
export DEVICE_BATCH_SIZE=8
export DATASET_SHARDS=1000
export DATASET_WORKERS=8
export NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-7}"

bash runs/train_4b.sh
```

If Slurm grants a non-H200 7-GPU node, cancel and resubmit more selectively. If the cluster cannot provide 7 H200s, use 5 H200s on `watgpu-800` or 4 H200s on `watgpu-700`; the code now adjusts the auto batch size to non-power-of-two GPU counts.

## Corpus downloads

`runs/train_4b.sh` downloads:

- ClimbMix parquet shards from `karpathy/climbmix-400b-shuffle`.
- The tokenizer training subset.
- The CORE eval bundle.
- The identity conversation JSONL.
- SFT and chat-eval datasets: SmolTalk, MMLU, GSM8K, ARC, HumanEval, and the spelling word list.

Default `DATASET_SHARDS=1000` downloads enough ClimbMix for the d36 ratio-12 run with room to spare. Set `DATASET_SHARDS=-1` only if you want to mirror the full hosted ClimbMix parquet corpus; that is roughly 600 GB.

## Useful overrides

```bash
WANDB_RUN=d36_4b_h200 bash runs/train_4b.sh
NPROC_PER_NODE=4 DEVICE_BATCH_SIZE=4 bash runs/train_4b.sh
DATASET_SHARDS=-1 bash runs/train_4b.sh
SKIP_SETUP=1 bash runs/train_4b.sh
```

For a fast allocation smoke test:

```bash
STOP_AFTER=base_train \
EXTRA_BASE_TRAIN_ARGS="--num-iterations=2 --save-every=-1 --core-metric-every=-1 --sample-every=-1 --eval-every=-1" \
bash runs/train_4b.sh
```
