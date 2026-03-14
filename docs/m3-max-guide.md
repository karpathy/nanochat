---
title: "Phase 1.5 on Apple Silicon M3 Max"
summary: "Practical guide for running compression-based optimization experiments on M3 Max with 128GB memory"
read_when: "Running Phase 1.5 experiments on Apple Silicon hardware"
status: draft
last_updated: 2026-03-05
---

# Phase 1.5 on Apple Silicon M3 Max (128GB)

## Hardware Capabilities

**Your Setup**: M3 Max with 128GB unified memory
- **Excellent for**: Small to medium model experiments (d8-d16)
- **Feasible for**: Larger models (d20-d24) with optimizations
- **Perfect for**: Phase 1.5 validation experiments

## What You Can Do

### ✅ Fully Feasible Experiments

#### 1. Compression Metrics Tracking (Phase 1.5.1)

**Model scales**: d8, d12, d16 (comfortable), d20 (tight)

```bash
# Optimal: d12 model with compression tracking
python -m scripts.base_train \
    --depth=12 \
    --track-compression \
    --compression-log-every=100 \
    --device=mps \
    --batch-size=8 \
    --sequence-len=1024
```

**Why this works well**:
- Compression metrics are CPU-friendly
- Apple's Accelerate framework excellent for SVD/entropy
- 128GB plenty for d12-d16 + compression overhead
- Can run overnight experiments easily

#### 2. Dataset Compression Evaluation (Phase 3.1)

**Fully feasible** - this is mostly CPU work:

```python
# Evaluate datasets without training
python -m scripts.evaluate_dataset_compression \
    --datasets climbmix fineweb dclm dolma3 \
    --sample-size=1000000000  # 1B tokens
    --output=dataset_rankings.json
```

**Why this is perfect**:
- No GPU needed for dataset evaluation
- Compression algorithms (gzip, entropy) are CPU-bound
- Can evaluate multiple datasets in parallel
- 128GB memory handles large dataset samples easily

#### 3. Small-Scale Validation Experiments

**Train multiple d12 models to validate compression predictions**:

```bash
# Experiment 1: Baseline
python -m scripts.base_train --depth=12 --dataset=climbmix --track-compression

# Experiment 2: Different dataset
python -m scripts.base_train --depth=12 --dataset=fineweb --track-compression

# Experiment 3: Compression-aware optimizer
python -m scripts.base_train --depth=12 --optimizer=compression-aware-muon
```

**Timeline**: Each d12 run takes ~12-24 hours on M3 Max
- Can run 3-4 experiments per week
- 4 weeks = 12-16 validation experiments
- Enough to validate compression approach

### ⚠️ Feasible with Optimizations

#### 4. d16-d20 Models (Reduced Batch Size)

```bash
# d16 with smaller batch size
python -m scripts.base_train \
    --depth=16 \
    --batch-size=4 \
    --gradient-accumulation-steps=4 \
    --track-compression \
    --device=mps
```

**Memory optimization strategies**:
- Reduce batch size (4-8 instead of 16-32)
- Use gradient accumulation to maintain effective batch size
- Reduce sequence length (1024 instead of 2048)
- Enable gradient checkpointing if needed

#### 5. Compression-Aware Optimizer (Phase 1.5.3)

**Feasible but slower**:

```python
# Compression-aware Muon on d12
python -m scripts.base_train \
    --depth=12 \
    --optimizer=compression-aware-muon \
    --compression-threshold=0.8 \
    --compression-update-every=1000  # Less frequent updates
```

**Why slower**:
- SVD computation for compression values
- But only every 1000 steps, so overhead is manageable
- Apple's Accelerate framework makes SVD fast

### ❌ Not Feasible

#### 6. d24-d26 Full Training

**Too large for comfortable training**:
- d24 (600M params) needs ~40-50GB
- Leaves limited memory for batch processing
- Training would be very slow
- Better to validate on d12-d16, then use cloud for d24+

#### 7. Multi-GPU Experiments

**M3 Max is single-device**:
- No distributed training
- But Phase 1.5 doesn't require it!
- Validation experiments work fine on single device

## Recommended Experimental Plan

### Week 1-2: Implement Compression Metrics

```bash
# Step 1: Add compression tracking to base_train.py
# (Use code from phase-1.5-compression-optimization.md)

# Step 2: Test on tiny model first
python -m scripts.base_train \
    --depth=8 \
    --max-steps=1000 \
    --track-compression \
    --device=mps

# Step 3: Validate metrics make sense
# Check wandb logs for compression_ratio, entropy, etc.
```

### Week 3-4: Dataset Compression Evaluation

```python
# Evaluate 5 major datasets
datasets = [
    'karpathy/climbmix-400b-shuffle',
    'karpathy/fineweb-edu-100b-shuffle',
    'mlfoundations/dclm-baseline-1.0',
    'allenai/dolma3_mix-6T',
    'bigcode/starcoderdata',
]

for dataset in datasets:
    # Sample 1B tokens (fast, ~1-2 hours per dataset)
    score = evaluate_compression_quality(dataset, sample_size=1e9)
    print(f"{dataset}: {score}")

# Expected output:
# climbmix: 2.45 (highest - explains 27% speedup!)
# fineweb: 2.31
# dclm: 2.38
# dolma3: 2.28
# starcoder: 2.52 (code is highly compressible)
```

### Week 5-8: Validation Experiments (d12)

```bash
# Train d12 on top 3 datasets
for dataset in climbmix fineweb dclm; do
    python -m scripts.base_train \
        --depth=12 \
        --dataset=$dataset \
        --max-steps=10000 \
        --track-compression \
        --device=mps \
        --wandb-project=phase15-validation
done

# Each run: ~18-24 hours on M3 Max
# Total: ~3-4 days of compute
# Can run overnight + weekends
```

### Week 9-12: Compression-Aware Optimizer

```bash
# Baseline: Standard Muon
python -m scripts.base_train \
    --depth=12 \
    --optimizer=muon \
    --track-compression \
    --wandb-run-name=baseline-muon

# Experiment: Compression-Aware Muon  
python -m scripts.base_train \
    --depth=12 \
    --optimizer=compression-aware-muon \
    --compression-threshold=0.8 \
    --track-compression \
    --wandb-run-name=compression-aware-muon

# Compare convergence speed and final loss
```

## M3 Max Specific Optimizations

### 1. MPS Backend Configuration

```python
# In base_train.py, add MPS support
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS backend")
else:
    device = torch.device("cpu")

# MPS-specific optimizations
if device.type == "mps":
    # Disable some features that don't work well on MPS
    torch.backends.mps.enable_fallback = True
```

### 2. Memory Management

```python
# Aggressive memory cleanup for M3 Max
import gc

def cleanup_memory():
    """Free memory on M3 Max"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Call after each validation step
if step % args.eval_every == 0:
    evaluate(model, val_loader)
    cleanup_memory()
```

### 3. Batch Size Tuning

```python
# Auto-tune batch size for M3 Max
def find_optimal_batch_size(model, device, max_memory_gb=100):
    """Find largest batch size that fits in memory"""
    batch_size = 32
    while batch_size >= 1:
        try:
            # Test forward + backward pass
            dummy_input = torch.randint(0, 32000, (batch_size, 1024)).to(device)
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            print(f"Batch size {batch_size} fits!")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                cleanup_memory()
            else:
                raise
    
    return 1

# Use in training script
optimal_batch_size = find_optimal_batch_size(model, device)
print(f"Using batch size: {optimal_batch_size}")
```

### 4. Compression Metrics Optimization

```python
# Optimize compression metrics for M3 Max
class CompressionMetrics:
    def compute_compression_ratio(self, tokens, logits):
        # Move to CPU for compression computation
        # (CPU is fast enough, saves MPS memory)
        tokens_cpu = tokens.cpu()
        logits_cpu = logits.cpu()
        
        # Compute on CPU using Accelerate framework
        entropy = self.compute_entropy(tokens_cpu)
        conditional_entropy = self.compute_conditional_entropy(
            tokens_cpu, logits_cpu
        )
        
        return entropy / conditional_entropy
```

## Expected Performance

### Training Speed (M3 Max vs H100)

| Model | M3 Max (128GB) | H100 (80GB) | Ratio |
|-------|----------------|-------------|-------|
| d8    | ~8K tok/sec    | ~50K tok/sec | 6x slower |
| d12   | ~4K tok/sec    | ~30K tok/sec | 7.5x slower |
| d16   | ~2K tok/sec    | ~20K tok/sec | 10x slower |
| d20   | ~1K tok/sec    | ~15K tok/sec | 15x slower |

**But for Phase 1.5 validation, this is fine!**
- d12 to 10K steps: ~2.5 hours on M3 Max vs ~20 min on H100
- Still feasible for overnight experiments
- Validation doesn't require full training runs

### Memory Usage

| Model | Params | Memory (fp32) | Memory (bf16) | Fits in 128GB? |
|-------|--------|---------------|---------------|----------------|
| d8    | 42M    | ~4GB          | ~2GB          | ✅ Easily |
| d12   | 110M   | ~10GB         | ~5GB          | ✅ Comfortably |
| d16   | 235M   | ~20GB         | ~10GB         | ✅ Yes |
| d20   | 400M   | ~35GB         | ~18GB         | ✅ Tight but OK |
| d24   | 600M   | ~50GB         | ~25GB         | ⚠️ Very tight |

**With 128GB, you have plenty of headroom for d12-d16 experiments.**

## What You'll Learn (Same as H100)

### Validation Questions Answered

✅ **Does compression ratio correlate with performance?**
- Train d12 on 3 datasets, compare compression scores
- Answer: Yes/No with R² correlation

✅ **Can we predict dataset quality before training?**
- Evaluate 5 datasets, train on top 3
- Answer: Compression score predicts performance

✅ **Does compression-aware optimizer improve convergence?**
- Compare standard Muon vs compression-aware Muon
- Answer: X% faster convergence (or no difference)

✅ **Do compression metrics detect overfitting early?**
- Track compression plateau vs loss plateau
- Answer: Compression plateaus X steps before loss

### Decision Point (Same as H100)

After 3-4 months of experiments on M3 Max:

**If >15% improvement**:
- Compression approach validated ✅
- Rent cloud GPUs for Phase 2 (infrastructure)
- Scale to d24-d26 on H100s
- You know it's worth the investment

**If <5% improvement**:
- Compression approach doesn't work ❌
- Don't waste money on cloud GPUs
- Pivot to Phase 6 (SP-Transformer hybrid)
- Continue research on M3 Max

**If 5-15% improvement**:
- Mixed results, need refinement
- Continue iterating on M3 Max
- More targeted experiments
- Re-evaluate after refinement

## Cost Analysis

### M3 Max (Your Hardware)

**Cost**: $0 (already own it)
**Time**: 3-4 months
**Experiments**: 12-16 validation runs
**Outcome**: Know if compression approach works

### Cloud Alternative (H100)

**Cost**: ~$2-3/hour × 24 hours × 30 days × 3 months = ~$5,000-7,000
**Time**: 3-4 months (same experiments, just faster)
**Experiments**: Same 12-16 runs, but 7x faster
**Outcome**: Same validation, but expensive

### Recommendation

**Use M3 Max for Phase 1.5 validation**:
- Zero cost
- Plenty of memory (128GB)
- Fast enough for d12-d16 experiments
- Can run overnight/weekend experiments
- Only rent cloud GPUs if validation succeeds

**Rent H100s only for**:
- Phase 2 (infrastructure) - requires multi-GPU
- d24-d26 training - too slow on M3 Max
- Final 7B training - requires distributed setup

## Practical Tips

### 1. Use Overnight Experiments

```bash
# Start before bed
nohup python -m scripts.base_train \
    --depth=12 \
    --max-steps=10000 \
    --track-compression \
    > experiment.log 2>&1 &

# Check progress in morning
tail -f experiment.log
```

### 2. Parallel Dataset Evaluation

```bash
# Evaluate multiple datasets in parallel (CPU-bound)
for dataset in climbmix fineweb dclm dolma3; do
    python -m scripts.evaluate_dataset_compression \
        --dataset=$dataset \
        --output=${dataset}_score.json &
done
wait

# All 4 datasets evaluated in ~2-3 hours
```

### 3. Checkpoint Frequently

```python
# Save checkpoints every 1000 steps (in case of crash)
if step % 1000 == 0:
    save_checkpoint(model, optimizer, step, f"checkpoint_{step}.pt")
```

### 4. Monitor Memory Usage

```bash
# Watch memory usage during training
watch -n 1 'ps aux | grep python | grep base_train'

# Or use Activity Monitor on macOS
```

## Conclusion

**Yes, your M3 Max (128GB) is perfect for Phase 1.5!**

✅ **What works**: All validation experiments (d8-d16)
✅ **What's feasible**: d20 with optimizations
❌ **What doesn't**: d24+ full training, multi-GPU

**Strategy**:
1. Validate compression approach on M3 Max (3-4 months, $0)
2. If successful (>15% improvement), rent H100s for scaling
3. If unsuccessful (<5%), continue research on M3 Max

**Bottom line**: You can complete all Phase 1.5 validation experiments on your M3 Max and make an informed decision about cloud GPU investment. The hardware you have is ideal for this phase.

---

*Your M3 Max with 128GB is actually better suited for Phase 1.5 validation than a single H100 (80GB), because you have more memory for experiments and zero cost.*
