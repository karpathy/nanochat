# nanochat UPPMAX Checkpoint Guide

This guide covers the improved checkpoint system for long-running nanochat training jobs on UPPMAX that automatically saves progress and enables resuming from failures.

## 🚀 Quick Start

### Start a new training run with checkpoints:
```bash
# Depth 20 (~12-15 hours, checkpoints every 2 hours)
sbatch runs/uppmax/train_d20_checkpoint.sh

# Depth 24 (~24-30 hours, checkpoints every 1.5 hours)  
sbatch runs/uppmax/train_d24_checkpoint.sh
```

### Resume from latest checkpoint:
```bash
# Automatically finds and resumes from the latest checkpoint
./runs/uppmax/resume_latest.sh 20  # for d20
./runs/uppmax/resume_latest.sh 24  # for d24
```

### Monitor training progress:
```bash
./runs/uppmax/monitor_training.sh 20  # for d20
./runs/uppmax/monitor_training.sh 24  # for d24
```

## 📁 Checkpoint System

### How it works:
- **Automatic checkpointing**: Saves model, optimizer, and metadata every N steps
- **Resume capability**: Can restart from any checkpoint if job fails or times out
- **Storage efficient**: Keeps all checkpoints (you can clean old ones manually if needed)
- **Progress tracking**: Maintains training statistics across restarts

### Checkpoint frequency:
- **d20**: Every 500 steps (~2 hours)
- **d24**: Every 400 steps (~1.5 hours)

### File structure:
```
~/.cache/nanochat/base_checkpoints/d20/
├── model_000500.pt      # Model weights at step 500
├── meta_000500.json     # Training metadata at step 500
├── optim_000500_rank0.pt # Optimizer state at step 500
├── model_001000.pt      # Model weights at step 1000
├── meta_001000.json     # Training metadata at step 1000
└── ...
```

## 🛠️ Usage Examples

### Example 1: Start fresh d20 training
```bash
sbatch runs/uppmax/train_d20_checkpoint.sh
```

### Example 2: Resume d20 from specific step
```bash
# Resume from step 2000
sbatch runs/uppmax/train_d20_checkpoint.sh 2000
```

### Example 3: Resume from latest automatically
```bash
./runs/uppmax/resume_latest.sh 20
```

### Example 4: Monitor progress
```bash
./runs/uppmax/monitor_training.sh 20
```

## 📊 Monitoring Training

### Check job status:
```bash
squeue -u $USER
```

### View live logs:
```bash
tail -f ~/nanochat-d20-ckpt-*.out
```

### Check latest checkpoint:
```bash
./runs/uppmax/monitor_training.sh 20
```

### View training metrics:
- Check wandb dashboard if configured
- Look at `val_bpb` in checkpoint metadata
- Monitor `core_metric` for GPT-2 level performance

## 🔧 Troubleshooting

### Job timed out at 24 hours?
```bash
# Just resume from latest checkpoint
./runs/uppmax/resume_latest.sh 20
```

### Out of storage space?
```bash
# Check usage
du -sh ~/.cache/nanochat

# Clean old checkpoints (keep every 5th)
cd ~/.cache/nanochat/base_checkpoints/d20
ls model_*.pt | sed -n '1~5!p' | xargs rm -f
ls meta_*.json | sed -n '1~5!p' | xargs rm -f  
ls optim_*.pt | sed -n '1~5!p' | xargs rm -f
```

### Training crashed?
```bash
# Check the error log
cat ~/nanochat-d20-ckpt-*.err

# Resume from latest good checkpoint
./runs/uppmax/resume_latest.sh 20
```

### Can't find checkpoints?
```bash
# Check if checkpoint directory exists
ls -la ~/.cache/nanochat/base_checkpoints/

# Verify step numbers
ls ~/.cache/nanochat/base_checkpoints/d20/model_*.pt
```

## ⚙️ Configuration

### Adjust checkpoint frequency:
Edit the `--save-every` parameter in the training scripts:
- More frequent: `--save-every=250` (saves more often, uses more storage)
- Less frequent: `--save-every=1000` (saves less often, less storage, higher risk)

### Modify batch size for memory:
If you get OOM errors, reduce `--device-batch-size`:
- d20: Try `--device-batch-size=6` or `--device-batch-size=4`
- d24: Try `--device-batch-size=4` or `--device-batch-size=3`

### Change time limits:
Modify the `#SBATCH -t` directive in the scripts:
- Short test: `#SBATCH -t 4:00:00` (4 hours)
- Long run: `#SBATCH -t 48:00:00` (48 hours)

## 🎯 Training Strategy

### For d20 (GPT-1 level):
- Expected time: ~12-15 hours
- Checkpoint every 2 hours (500 steps)
- Target: val_bpb < 0.75, core_metric > 0.25

### For d24 (approaching GPT-2 level):
- Expected time: ~24-30 hours  
- Checkpoint every 1.5 hours (400 steps)
- Target: val_bpb < 0.74, core_metric > 0.256

### Resume strategy:
1. Let job run to completion if possible
2. If job times out, immediately resume with `resume_latest.sh`
3. Monitor progress with `monitor_training.sh`
4. Continue until target metrics are reached

## 📈 Performance Tips

### Optimize for L40s:
- Use batch sizes that fully utilize 48GB VRAM
- Monitor GPU utilization in logs
- Adjust `--device-batch-size` if needed

### Storage optimization:
- Clean old checkpoints periodically
- Use `~/.cache/nanochat` for all training data
- Monitor storage with `uquota`

### Wandb integration:
- Save API key in `~/.wandb_key` for logging
- Monitor training curves in real-time
- Compare runs across different configurations

This checkpoint system makes long nanochat training runs much more reliable on UPPMAX! 🚀