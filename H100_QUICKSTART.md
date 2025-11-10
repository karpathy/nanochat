# Quick Start Guide: Running nanochat on H100 GPUs

## TL;DR - The Fix Is Already Applied! 🎉

If you're seeing this after pulling the latest changes, **you don't need to do anything special**. The H100 CUDA error fix (Issue #257) is automatically applied when you run:

```bash
bash speedrun.sh
```

## What Was Fixed?

Previously, running `speedrun.sh` on H100 GPUs would crash at step 0 with:
```
torch.AcceleratorError: CUDA error: invalid argument
```

This has been fixed by disabling aggressive Triton kernel autotuning that was causing invalid tensor configurations.

## Running on H100 (Lambda.ai or Similar)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Run speedrun (the fix is automatic)
bash speedrun.sh
```

### 2. Monitor Progress

The training will run for ~4 hours. You should see:

```
Step 00000 | Validation bpb: 3.3013
step 00001/10000 (0.01%) | loss: 10.123456 | ...
step 00002/10000 (0.02%) | loss: 10.098765 | ...
```

✅ **Success**: Training proceeds past step 0 without CUDA errors

### 3. Optional: Run in Screen Session

Since training takes 4 hours, use screen:

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

Detach with `Ctrl-a d`, reattach with `screen -r speedrun`

## Verification (Optional)

Want to verify the fix before running the full 4-hour training?

### Quick Test (5 iterations)

```bash
# Setup environment
source .venv/bin/activate

# Test with 5 iterations only
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --num_iterations=5
```

This should complete in ~1 minute and verify:
- ✅ Model compiles successfully
- ✅ Forward pass works
- ✅ Backward pass works (where the error occurred)
- ✅ Training can proceed

### Standalone Test Script

```bash
python test_h100_fix.py
```

## What Changed Under the Hood?

The fix adds these environment variables before PyTorch operations:

```python
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "0"
```

These are set in:
- `speedrun.sh` (global export)
- `scripts/base_train.py`
- `scripts/mid_train.py`
- `scripts/chat_sft.py`

## Performance Impact

**Expected**: 5-10% reduction in throughput

**Why it's OK**:
- Training completes instead of crashing ✅
- MFU still 40-60% on H100 ✅
- Total cost still ~$100 for speedrun ✅

## Troubleshooting

### Still Getting CUDA Errors?

1. **Check PyTorch version**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   Should be 2.8.0 or later

2. **Check CUDA version**:
   ```bash
   nvidia-smi
   ```
   Should show CUDA 12.8 or compatible

3. **Try disabling compilation** (in `scripts/base_train.py`):
   ```python
   # model = torch.compile(model, dynamic=False)  # Comment this out
   ```

4. **Report on GitHub Issue #257** with:
   - GPU model
   - CUDA version
   - PyTorch version
   - Full error traceback

### Out of Memory?

Reduce batch size in speedrun.sh:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=16
```

(Default is 32, try 16, 8, or 4)

## Expected Results

After ~4 hours, you should have:

- ✅ Trained d20 model (561M parameters)
- ✅ Validation bpb: ~2.5-3.0
- ✅ CORE metric: ~0.22
- ✅ Model checkpoint in `~/.cache/nanochat/base_checkpoints/d20/`
- ✅ Report card in `report.md`

## Next Steps

After training completes:

### Chat with Your Model

```bash
# Web UI (recommended)
python -m scripts.chat_web

# CLI
python -m scripts.chat_cli -p "Why is the sky blue?"
```

### View Report Card

```bash
cat report.md
```

### Train Larger Models

See README.md for d26 ($300) and d32 ($1000) configurations.

## System Requirements

- **GPU**: 8x H100 (80GB each) or 8x A100
- **RAM**: 256GB+ recommended
- **Storage**: ~30GB for data + checkpoints
- **Time**: ~4 hours for speedrun
- **Cost**: ~$100 on Lambda.ai ($24/hr for 8xH100)

## Support

- **Documentation**: See `ISSUE_257_FIX.md` for detailed technical info
- **GitHub Issue**: #257
- **General Questions**: GitHub Discussions

---

**Happy Training! 🚀**

The nanochat team
