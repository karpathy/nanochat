# Quick Start Guide: C++ Inference with nanochat

This guide will get you up and running with C++ inference in under 10 minutes.

## Prerequisites

Choose one of the following:

### Option A: LibTorch (TorchScript)

1. Download LibTorch from https://pytorch.org/get-started/locally/
2. Extract to a location (e.g., `/opt/libtorch`)
3. Set environment variable:
   ```bash
   export CMAKE_PREFIX_PATH=/opt/libtorch
   ```

### Option B: ONNX Runtime

1. Download from https://github.com/microsoft/onnxruntime/releases
2. Extract to a location (e.g., `/opt/onnxruntime`)
3. Set environment variable:
   ```bash
   export ONNXRUNTIME_DIR=/opt/onnxruntime
   ```

## Step 1: Export Your Model

From the nanochat root directory:

```bash
# For LibTorch
python -m scripts.export_model --source sft --format torchscript --output model.pt

# For ONNX Runtime
python -m scripts.export_model --source sft --format onnx --output model.onnx
```

This will create `model.pt` or `model.onnx` in the current directory.

## Step 2: Build the C++ Example

```bash
cd examples/cpp_inference
mkdir build && cd build

# For LibTorch only
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch -DBUILD_ONNX_EXAMPLE=OFF ..

# For ONNX Runtime only
cmake -DONNXRUNTIME_DIR=/opt/onnxruntime -DBUILD_LIBTORCH_EXAMPLE=OFF ..

# For both
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch -DONNXRUNTIME_DIR=/opt/onnxruntime ..

# Build
make -j$(nproc)
```

## Step 3: Run Inference

```bash
# LibTorch (CPU)
./libtorch_inference ../../../model.pt

# LibTorch (CUDA)
./libtorch_inference ../../../model.pt 1

# ONNX Runtime (CPU)
./onnx_inference ../../../model.onnx

# ONNX Runtime (CUDA)
./onnx_inference ../../../model.onnx 1
```

## Expected Output

```
Loading model from: model.pt
✓ Model loaded successfully

Prompt token IDs: 1 464 11742 15150 315 3090 374

--- Single Forward Pass ---
Output shape: [1, 7, 50304]
Next token (greedy): 473

--- Autoregressive Generation ---
Generating 20 tokens...
  Generated 10/20 tokens
  Generated 20/20 tokens

Generated token IDs: 1 464 11742 15150 315 3090 374 473 ...

✓ Inference completed successfully!
```

## Next Steps

### 1. Tokenization

The examples use hardcoded token IDs. To use real text:

**Option A: Python Tokenization**

```python
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init

device_type = "cpu"
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("sft", device, phase="eval")

# Encode
text = "Hello, how are you?"
bos = tokenizer.get_bos_token_id()
tokens = tokenizer.encode(text, prepend=bos)
print(tokens)  # Use these in C++

# Decode
generated_tokens = [1, 464, 11742, ...]
text = tokenizer.decode(generated_tokens)
print(text)
```

**Option B: C++ Tokenization**

Implement a BPE tokenizer in C++ using the vocabulary file. The nanochat tokenizer is tiktoken-compatible.

### 2. Customize Generation

Modify the C++ code to adjust:

- `temperature`: Controls randomness (0.0 = greedy, 1.0 = default, 2.0 = very random)
- `top_k`: Limits sampling to top-k tokens (50 is a good default)
- `max_tokens`: Maximum number of tokens to generate

### 3. Production Deployment

For production use:

1. **Implement KV Caching**: Use `ExportableGPTWithCache` for faster generation
2. **Batch Processing**: Modify code to process multiple sequences in parallel
3. **Error Handling**: Add robust error handling and logging
4. **Model Quantization**: Consider INT8/FP16 quantization for faster inference

## Troubleshooting

### "libtorch not found"

Make sure `CMAKE_PREFIX_PATH` points to the LibTorch directory:
```bash
export CMAKE_PREFIX_PATH=/path/to/libtorch
```

### "onnxruntime not found"

Make sure `ONNXRUNTIME_DIR` is set:
```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

### "Model loading failed"

Verify the model was exported successfully:
```bash
python -m scripts.export_model --source sft --format torchscript --output test.pt
```

### "Out of memory"

Reduce batch size or use CPU instead of GPU:
```bash
./libtorch_inference model.pt 0  # Use CPU
```

## Performance Tips

1. **Use CUDA**: GPU inference is 10-100x faster than CPU
2. **Optimize Batch Size**: Process multiple sequences together
3. **Use KV Cache**: Avoid recomputing past tokens
4. **Quantize Models**: INT8 quantization can provide 2-4x speedup

## Getting Help

- See [README.md](README.md) for detailed documentation
- Check [EXPORT_IMPLEMENTATION.md](../../EXPORT_IMPLEMENTATION.md) for implementation details
- Open an issue on GitHub for bugs or questions

## Example: Complete Workflow

```bash
# 1. Train a model (or use existing)
cd /path/to/nanochat
bash speedrun.sh

# 2. Export the model
python -m scripts.export_model --source sft --format torchscript --output model.pt

# 3. Build C++ example
cd examples/cpp_inference
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
make

# 4. Run inference
./libtorch_inference ../../../model.pt 1

# 5. Integrate into your application
# Copy the inference code into your project and customize as needed
```

That's it! You now have a working C++ inference setup for nanochat models.
