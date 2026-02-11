# nanochat on UPPMAX Pelle

Scripts for running nanochat on UPPMAX's Pelle GPU cluster.

## Quick Start

1. **Clone and setup** (on login node):
```bash
cd ~
git clone https://github.com/BirgerMoell/nanochat.git
cd nanochat
bash runs/uppmax/setup.sh
```

2. **Submit test job**:
```bash
sbatch runs/uppmax/train_test.sh
```

3. **Monitor**:
```bash
squeue -u $USER
tail -f nanochat-*.out
```

## Scripts

- `setup.sh` - One-time environment setup (run on login node)
- `train_test.sh` - Quick test with depth=8 (~5-10 min on L40s)
- `train_small.sh` - Small model with depth=12 (~30 min on L40s)
- `train_full.sh` - Full GPT-2 grade model (requires 8xH100)

## GPU Options

For Pelle, you can request:
- `--gpus=l40s:1` - L40s (48GB VRAM) - good for depth up to ~20
- `--gpus=h100:1` - H100 (80GB VRAM) - for larger models

## Notes

- Project: `uppmax2025-2-290`
- Max job time on GPU partition: 2 days
- Cache/checkpoints go to `~/.cache/nanochat`
