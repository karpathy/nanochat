# WATGPU handoff for d36 4B-class nanochat

This repo is now set up for an end-to-end d36 ClimbMix run through:

```bash
bash runs/train_4b.sh
```

The script trains a d36 model, about 3.8B parameters, on ClimbMix, then runs base eval, SFT, chat eval, and report generation. It also downloads the needed corpora and eval data into `NANOCHAT_BASE_DIR`.
On WATGPU, prefer submitting the checked-in Slurm wrapper:

```bash
sbatch runs/watgpu_4b.sbatch
```

By default this stores run artifacts and checkpoints in `/u201/l39gu/nanoknow-climbmix/nanochat-runs/d36_4b_climbmix`, uses the local ClimbMix mirror at `/u201/l39gu/projects/climbmix-400b-shuffle`, checkpoints every 250 optimizer steps, keeps the latest 4 checkpoints, and automatically resumes from the latest base checkpoint on repeat submissions.

The `/u201/l39gu` filesystem is mounted `noexec` on the login node, so native Python wheels cannot be loaded from a venv stored there. `runs/train_4b.sh` therefore creates its uv environment under `$SLURM_TMPDIR/nanochat-venv` when available, otherwise `/tmp/$USER/nanochat-venv`. Override `NANOCHAT_ENV_DIR` only with a path on an executable filesystem.

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
#SBATCH --signal=B:USR1@900
#SBATCH -o nanochat4b-%j.out
#SBATCH -e nanochat4b-%j.err

set -euo pipefail
cd /path/to/nanochat

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-4b}"
export CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-/u201/l39gu/nanoknow-climbmix/corpus/climbmix-400b-shuffle}"
export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix}"
export MODEL_DEPTH=36
export TARGET_PARAM_DATA_RATIO=12
export DEVICE_BATCH_SIZE=8
export DATASET_SHARDS=1000
export DATASET_WORKERS=8
export NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-7}"
export SAVE_EVERY=500
export AUTO_RESUME=1

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

Default `DATASET_SHARDS=1000` downloads enough ClimbMix for the d36 ratio-12 run with room to spare. Set `CLIMBMIX_DATA_DIR` to reuse a local mirror instead of copying shards into each run directory. Set `DATASET_SHARDS=-1` only if you want to mirror the full hosted ClimbMix parquet corpus; that is roughly 600 GB.

The local `/u201/l39gu/nanoknow-climbmix/corpus/climbmix-400b-shuffle` mirror already has more than enough train shards for the default d36 run. The repo still expects the validation shard `shard_06542.parquet`; `runs/train_4b.sh` will download it into `CLIMBMIX_DATA_DIR` if it is not already present.

## Checkpoint and resume

Base checkpoints are written under:

```bash
$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/
```

Each checkpoint contains model parameters, rank-local optimizer state, metadata, and dataloader position. Resume from any saved base checkpoint with:

```bash
RESUME_FROM_STEP=12345 bash runs/train_4b.sh
```

For batch jobs, keep `AUTO_RESUME=1` and reuse the same `NANOCHAT_BASE_DIR`; the script will find the latest `model_*.pt` checkpoint and pass the matching `--resume-from-step` automatically. The Slurm wrapper asks for a 7-day allocation and requests a `USR1` signal 15 minutes before the time limit so training can checkpoint and exit cleanly.

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
EXTRA_BASE_TRAIN_ARGS="--num-iterations=2 --save-every=1 --core-metric-every=-1 --sample-every=-1 --eval-every=-1" \
bash runs/train_4b.sh
```
