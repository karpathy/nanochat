# Inference Utilities

This package provides efficient inference and benchmarking tools for UNet models.

## Components

### 1. `unet_engine.py` - KV-Cached Inference Engine

Efficient inference engine for UNet models with KV caching support.

**Features:**
- Hierarchical KV cache for encoder-decoder architecture
- Caches encoder outputs for skip connections
- Batch generation support
- Comparison with naive generation for correctness verification

**Usage:**
```python
from inference import UNetEngine

engine = UNetEngine(model, tokenizer)

# Generate single sample
for token_column, masks in engine.generate(prompt_tokens, num_samples=1, max_tokens=64):
    print(tokenizer.decode([token_column[0]]))

# Generate multiple samples in parallel
results, masks = engine.generate_batch(prompt_tokens, num_samples=4, max_tokens=64)
```

**Standalone test:**
```bash
# Compares naive vs cached generation (requires trained UNet checkpoint)
python -m inference.unet_engine
```

### 2. `benchmark.py` - Performance Benchmarking

Tools for measuring inference performance.

**Features:**
- Prefill latency and decode throughput measurement
- Memory usage tracking
- Sequence length scaling tests
- Batch size scaling tests
- Comparison between generation methods

**Usage:**
```python
from inference import InferenceBenchmark

benchmark = InferenceBenchmark(model, tokenizer, device, autocast_ctx)

# Compare naive vs engine
comparison = benchmark.compare_methods(
    prompt_tokens=tokens,
    max_tokens=64,
    engine=engine,
)
print(comparison)

# Sweep sequence lengths
results = benchmark.sweep_sequence_lengths(
    engine=engine,
    base_prompt="Hello world",
    lengths=[64, 128, 256, 512],
)
```

### 3. `profiler.py` - Detailed Profiling

Advanced profiling utilities using PyTorch profiler.

**Features:**
- PyTorch profiler integration
- CUDA kernel timing
- Layer-wise timing breakdown
- Memory profiling
- Chrome trace export for visualization

**Usage:**
```python
from inference import TorchProfiler, SimpleTimer, MemoryProfiler

# PyTorch profiler with auto-generated filename
profiler = TorchProfiler(device_type="cuda")
result = profiler.profile_function(
    my_inference_func,
    warmup_iters=3,
    profile_iters=10,
    auto_save=True,  # Auto-generates filename with timestamp and parameters
    prompt_length=128,
    max_tokens=64,
    batch_size=1,
    temperature=0.0,
    model_tag="d[6,12]-h[768,1536]",
)
# Trace saved to: base_checkpoints/d[6,12]-h[768,1536]/profiles/profile_20260128_143022_prompt128_gen64_t0p0.json

# Or use explicit path
with profiler.profile(output_path="trace.json") as prof:
    model(inputs)

# Simple timer
timer = SimpleTimer()
with timer.measure("forward"):
    model(inputs)
print(timer.summary())

# Memory profiler
mem = MemoryProfiler()
mem.snapshot("before")
model(inputs)
mem.snapshot("after")
print(mem.summary())
```

## Profiling Trace Files

Traces are automatically saved with informative filenames in the checkpoint directory:

**Location:** `base_checkpoints/<model_tag>/profiles/`

**Filename format:** `profile_<method>_<timestamp>_<parameters>.json`

**Examples:**
- `profile_naive_20260128_143022_prompt128_gen64_t0p0.json` (no KV cache)
- `profile_engine_20260128_143022_prompt128_gen64_t0p0.json` (with KV cache)

**Filename components:**
- `naive` or `engine` - Generation method used
- `20260128_143022` - Date and time (YYYYMMDD_HHMMSS)
- `prompt128` - Prompt length: 128 tokens
- `gen64` - Generated: 64 tokens
- `bs4` - Batch size: 4 (omitted if batch_size=1)
- `t0p8` - Temperature: 0.8 (uses 'p' instead of '.')

This naming scheme makes it easy to:
- Find traces by date/time
- Compare different configurations
- Organize experiments
- Compare naive vs engine performance side-by-side

### Profiling Modes

**Naive mode (default):**
- Uses `model.generate()` - no KV caching
- Recomputes all attention for each token
- Slower but simpler to understand in profiler
- Good for understanding model architecture

**Engine mode (`--profile-engine`):**
- Uses `UNetEngine.generate()` - with KV caching
- Caches key/value tensors, encoder outputs
- Much faster for generation
- Shows real-world optimized performance
- Good for production performance analysis

**Comparing both:**
```bash
# Profile naive
python -m scripts.benchmark_unet --profile \
    --model-tag d[6,12]-h[768,1536]
# → profile_naive_20260128_143022_prompt128_gen64_t0p0.json

# Profile engine  
python -m scripts.benchmark_unet --profile --profile-engine \
    --model-tag d[6,12]-h[768,1536]
# → profile_engine_20260128_143045_prompt128_gen64_t0p0.json

# Load both in chrome://tracing to compare!
```

## Command-Line Tools

### Benchmark UNet Inference

Comprehensive benchmarking script in `scripts/benchmark_unet.py`:

```bash
# Basic comparison (naive vs engine)
python -m scripts.benchmark_unet --compare

# Full benchmark suite
python -m scripts.benchmark_unet --all

# Profile naive generation (no KV cache) - default
python -m scripts.benchmark_unet --profile

# Profile engine generation (with KV cache) - optimized version
python -m scripts.benchmark_unet --profile --profile-engine

# Or specify custom output path
python -m scripts.benchmark_unet --profile --profile-output custom_trace.json

# Sequence length sweep
python -m scripts.benchmark_unet --sweep-lengths --lengths 64 128 256 512

# Batch size sweep  
python -m scripts.benchmark_unet --sweep-batch --batch-sizes 1 2 4 8

# Profile and benchmark together
python -m scripts.benchmark_unet --compare --profile \
    --model-tag d[6,12]-h[768,1536]
# Saves trace to: base_checkpoints/d[6,12]-h[768,1536]/profiles/profile_<timestamp>_<params>.json

# Custom prompt and generation length
python -m scripts.benchmark_unet --compare \
    --prompt "The capital of France is" \
    --max-tokens 128

# Specify model checkpoint
python -m scripts.benchmark_unet --compare \
    --model-tag d[6,12]-h[768,1536] \
    --step 10000

# Save results to JSON
python -m scripts.benchmark_unet --all --output-json results.json
```

## Expected Performance

For a typical UNet model (2-stage, ~100M params):

- **Speedup from KV caching**: 2-5x for generation (depends on sequence length)
- **Memory overhead**: ~100-500MB for KV cache (depends on batch size and sequence length)
- **Throughput**: 
  - Short sequences (64 tokens): 50-100 tok/s
  - Long sequences (512 tokens): 20-50 tok/s

## Requirements

- PyTorch >= 2.0
- CUDA (optional, for profiling and best performance)
- Trained UNet checkpoint

## Training a UNet Model

If you don't have a trained UNet checkpoint:

```bash
# Train a small UNet model
python -m scripts.unet.base_train \
    --depth 6 12 \
    --model-dim 768 1536 \
    --num-iterations 1000 \
    --device-batch-size 32
```

## Visualization

Chrome traces from profiling can be viewed at `chrome://tracing`:

1. Run benchmark with profiling: `python -m scripts.benchmark_unet --profile`
2. Note the auto-generated trace path (e.g., `base_checkpoints/d[6,12]-h[768,1536]/profiles/profile_20260128_143022_prompt128_gen64_t0p0.json`)
3. Open Chrome browser
4. Go to `chrome://tracing`
5. Click "Load" and select the trace file
6. Explore timeline, zoom, and inspect individual operations

All traces are automatically organized by model in the checkpoint directory for easy comparison.

## Architecture Notes

The UNet engine handles the unique challenges of UNet's hierarchical structure:

1. **Prefill Phase (T > 1)**:
   - Full encoder pass through all stages (with pooling)
   - Caches encoder outputs at each stage for skip connections
   - Full decoder pass with skip connections
   
2. **Generation Phase (T = 1)**:
   - Encoder pooling stops early (only stage 0 runs)
   - Uses cached encoder outputs from prefill for skip connections
   - Full decoder pass
   
3. **KV Cache Structure**:
   - Separate caches for each stage (different sequence lengths)
   - Tracks position in original (unpooled) sequence
   - Efficient prefill replication for multi-sample generation

## Troubleshooting

**Error: "Loaded model is not a UNet architecture"**
- Make sure you're loading a UNet checkpoint (trained with `scripts/unet/base_train.py`)
- UNet models have `.encoder` and `.decoder` attributes

**Low speedup from caching**
- Normal for very short sequences (< 32 tokens)
- KV cache setup overhead dominates for short generation
- Speedup increases with longer generation

**Out of memory**
- Reduce batch size
- Reduce max sequence length
- Use smaller model

**Profiler not working**
- Profiling requires CUDA
- Make sure PyTorch is built with profiler support
- Try reducing `--profile-iters`
