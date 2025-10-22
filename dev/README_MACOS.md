# macOS / MPS Training Guide

This guide explains how to train nanochat on Apple Silicon Macs with automatic memory optimization.

## Memory-Optimized Scripts

All scripts now auto-detect your system memory and optimize batch sizes accordingly:

### Performance Profiles

| Memory | device_batch_size | total_batch_size | Speed Boost | Recommended For |
|--------|-------------------|------------------|-------------|-----------------|
| **128GB+** | 16 | 4096 | 16× | M3 Max/Ultra, Mac Studio Ultra |
| **64GB** | 8 | 2048 | 8× | M2/M3 Max, Mac Studio Max |
| **32GB** | 4 | 1024 | 4× | M2/M3 Pro, MacBook Pro |
| **<32GB** | 1 | 512 | 1× | Base M1/M2/M3 |

## Quick Start Scripts

### 1. `runcpu.sh` - Quick Test (30 minutes)
Fast validation that everything works:
```bash
bash dev/runcpu.sh
```

**What it does:**
- Trains depth=4 model (37M params)
- 50 base iterations + 100 mid + 100 SFT
- Good for testing, not production quality

**Your 128GB Mac:** ~15-30 minutes (16× faster!)

### 2. `runmac_overnight.sh` - Production Quality (2-8 hours)
Full training for better results:
```bash
bash dev/runmac_overnight.sh
```

**What it does:**
- Trains depth=6 model (82M params)
- 500 base iterations + 150 mid + 150 SFT
- Downloads 50 data shards
- Production-quality chatbot

**Your 128GB Mac:** ~2-3 hours (vs 8-12 hours at batch_size=1)

## Manual Configuration

Override memory detection:
```bash
# Pretend you have 64GB (more conservative)
MEMORY_SIZE=64 bash dev/runcpu.sh

# Set specific batch sizes
DEVICE_BATCH_SIZE=8 TOTAL_BATCH_SIZE=2048 bash dev/runmac_overnight.sh

# Combine overrides
DEPTH=8 MEMORY_SIZE=128 BASE_ITERATIONS=1000 bash dev/runmac_overnight.sh
```

## Environment Variables

All scripts support these overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_SIZE` | auto-detect | System memory in GB |
| `DEVICE_BATCH_SIZE` | auto-calc | Sequences per device |
| `TOTAL_BATCH_SIZE` | auto-calc | Total batch size in tokens |
| `EVAL_TOKENS` | auto-calc | Tokens for evaluation |
| `SPLIT_TOKENS` | auto-calc | Tokens for loss eval |
| `DEPTH` | 6 (overnight), 4 (cpu) | Model depth (layers) |
| `BASE_ITERATIONS` | 500 (overnight), 50 (cpu) | Base training steps |
| `MID_ITERATIONS` | 150 (overnight), 100 (cpu) | Midtraining steps |
| `SFT_ITERATIONS` | 150 (overnight), 100 (cpu) | SFT steps |
| `DATA_SHARDS` | 50 (overnight), 4 (cpu) | Training data shards |

## Expected Training Times (128GB Mac)

### Quick Test (`runcpu.sh`)
- Data download: 1-2 min
- Tokenizer: 1-2 min
- Base training (50 iter): 3-5 min
- Midtraining (100 iter): 6-10 min
- SFT (100 iter): 6-10 min
- **Total: 15-30 minutes**

### Overnight (`runmac_overnight.sh`)
- Data download: 5-10 min
- Tokenizer: 1-2 min
- Base training (500 iter): 40-60 min
- Midtraining (150 iter): 20-30 min
- SFT (150 iter): 20-30 min
- **Total: 2-3 hours**

## Model Quality Expectations

### After `runcpu.sh` (quick)
- Forms basic sentences
- Limited coherence
- Frequent hallucinations
- Good for testing setup

### After `runmac_overnight.sh` (production)
- Complete sentences
- Better coherence
- Follows conversation structure
- Still makes mistakes (it's small!)
- Good for demos/learning

### For GPT-2 Quality
Would need depth=20-32, billions of tokens, and 8×H100 GPUs ($800-1000)

## Memory Usage Tips

**Monitor memory:**
```bash
# Real-time memory usage
sudo powermetrics --samplers smc -i 5000 -n 1 | grep -i memory

# Or use Activity Monitor
open -a "Activity Monitor"
```

**If you get OOM errors:**
```bash
# Reduce batch size manually
DEVICE_BATCH_SIZE=4 bash dev/runmac_overnight.sh

# Or reduce model size
DEPTH=4 bash dev/runmac_overnight.sh
```

**Optimal setup for your 128GB Mac:**
```bash
# Maximum performance (recommended)
bash dev/runmac_overnight.sh

# Or go even bigger if you want
DEPTH=8 BASE_ITERATIONS=1000 bash dev/runmac_overnight.sh
```

## Continuing Training After Interruption

### Use `continue_training.sh` (Recommended)

If training was interrupted or you want to continue from existing checkpoints:

```bash
bash dev/continue_training.sh
```

**What it does:**
- ✅ Checks for existing base/mid/sft checkpoints
- ✅ Automatically continues from where you left off
- ✅ Skips completed stages
- ✅ Matches model tags (d4, d6, d8) correctly
- ✅ Uses memory-optimized batch sizes

**Example scenarios:**

1. **Base training completed, but mid/sft interrupted:**
   ```
   Status:
     ✓ Base model: d8/step_001000
     ✗ Midtraining: Not found

   → Will run: Midtraining → SFT
   ```

2. **Base and mid complete, only need SFT:**
   ```
   Status:
     ✓ Base model: d8/step_001000
     ✓ Midtraining: d8/step_000150
     ✗ SFT: Not found

   → Will run: SFT only
   ```

3. **Everything complete:**
   ```
   Status:
     ✓ Base model: d8/step_001000
     ✓ Midtraining: d8/step_000150
     ✓ SFT: d8/step_000150

   🎉 All training stages complete!
   → Ready to chat!
   ```

### Manual Continuation

If you prefer manual control:

```bash
source .venv/bin/activate

# Continue midtraining from existing base model
python -m scripts.mid_train \
  --num_iterations=150 \
  --device_batch_size=16

# Continue SFT from existing mid model
python -m scripts.chat_sft \
  --num_iterations=150 \
  --device_batch_size=16

# Chat with the result
python -m scripts.chat_cli -i sft
```

## Troubleshooting

### Training Won't Start

**Error: `AssertionError: total_batch_size must be divisible by...`**

Fix: Ensure `total_batch_size` is divisible by `device_batch_size × max_seq_len`
```bash
# For max_seq_len=1024:
# device_batch_size=16 → total_batch_size=16384 (16 × 1024)
# device_batch_size=8  → total_batch_size=8192  (8 × 1024)
```

**Error: `split_tokens must be divisible by tokens_per_step`**

Fix: Pass `--device_batch_size` to base_loss:
```bash
python -m scripts.base_loss --device_batch_size=16 --split_tokens=16384
```

### Architecture Issues

**Running x86_64 Python on ARM64 Mac (Rosetta 2)**

Check your Python architecture:
```bash
file .venv/bin/python
# Should show: Mach-O 64-bit executable arm64
# Bad: Mach-O 64-bit executable x86_64
```

Fix: Recreate venv with native ARM64 Python:
```bash
rm -rf .venv
uv venv --python /opt/homebrew/opt/python@3.10/bin/python3.10
uv sync
maturin develop --release
```

**Performance impact:** Native ARM64 is ~2-3× faster than Rosetta 2!

### Memory & Performance Issues

**Script fails with memory errors:**
- Reduce `MEMORY_SIZE=64` or `DEVICE_BATCH_SIZE=8`
- Reduce `DEPTH=4`
- Close other applications

**Training is slow:**
- Check memory profile: `sysctl hw.memsize`
- Verify MPS: Check logs for "Autodetected device type: mps"
- Verify ARM64: `file .venv/bin/python` should show `arm64`
- Check CPU usage: Should be 80-100% on one core

**Chat responses are still poor:**
- Increase iterations: `BASE_ITERATIONS=1000 MID_ITERATIONS=300 SFT_ITERATIONS=300`
- Download more data: `DATA_SHARDS=100`
- Increase model size: `DEPTH=8` (needs more memory)

## Running in Background

**Screen (recommended):**
```bash
screen -S nanochat bash dev/runmac_overnight.sh
# Detach: Ctrl+A, D
# Reattach: screen -r nanochat
```

**nohup:**
```bash
nohup bash dev/runmac_overnight.sh > training.log 2>&1 &
tail -f training.log
```

## After Training

**Chat via CLI:**
```bash
python -m scripts.chat_cli -i sft
```

**Chat via Web UI:**
```bash
python -m scripts.chat_web -i sft
# Visit http://localhost:8000
```

**Check your report:**
```bash
cat report_overnight.md
# or
cat ~/.cache/nanochat/report/report.md
```

## Notes

- All MPS compatibility fixes are applied automatically
- torch.compile is disabled on MPS (not supported yet)
- BFloat16 is replaced with float32 on MPS
- Pinned memory optimizations disabled on MPS
- Training is slower than CUDA but much faster than CPU

Enjoy your locally-trained LLM! 🚀
