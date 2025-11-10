# nanochat C++ Inference Examples

This directory contains C++ examples for running inference with nanochat models exported to TorchScript and ONNX formats.

## Prerequisites

### For LibTorch (TorchScript) Example

1. **Download LibTorch**
   - Visit: https://pytorch.org/get-started/locally/
   - Select your platform and download the C++ distribution (LibTorch)
   - Extract to a location, e.g., `/opt/libtorch` or `C:\libtorch`

2. **Set CMAKE_PREFIX_PATH**
   ```bash
   export CMAKE_PREFIX_PATH=/path/to/libtorch
   ```

### For ONNX Runtime Example

1. **Download ONNX Runtime**
   - Visit: https://github.com/microsoft/onnxruntime/releases
   - Download the appropriate package for your platform
   - Extract to a location, e.g., `/opt/onnxruntime` or `C:\onnxruntime`

2. **Set ONNXRUNTIME_DIR**
   ```bash
   export ONNXRUNTIME_DIR=/path/to/onnxruntime
   ```

## Building

### Linux/macOS

```bash
# Create build directory
mkdir build && cd build

# Configure (LibTorch only)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Configure (ONNX Runtime only)
cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime -DBUILD_LIBTORCH_EXAMPLE=OFF ..

# Configure (both)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DONNXRUNTIME_DIR=/path/to/onnxruntime ..

# Build
cmake --build . --config Release

# Or use make directly
make -j$(nproc)
```

### Windows

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake -DCMAKE_PREFIX_PATH=C:\libtorch -DONNXRUNTIME_DIR=C:\onnxruntime ..

# Build
cmake --build . --config Release
```

## Exporting Models

Before running the C++ examples, you need to export your trained nanochat model:

### Export to TorchScript

```bash
# Export SFT model to TorchScript
python -m scripts.export_model --source sft --format torchscript --output model.pt

# Export with specific model tag
python -m scripts.export_model --source mid --model-tag d20 --format torchscript --output model_d20.pt
```

### Export to ONNX

```bash
# Export SFT model to ONNX
python -m scripts.export_model --source sft --format onnx --output model.onnx

# Export both formats at once
python -m scripts.export_model --source sft --format both
```

## Running

### LibTorch Example

```bash
# CPU inference
./libtorch_inference /path/to/model.pt

# CUDA inference (if available)
./libtorch_inference /path/to/model.pt 1
```

### ONNX Runtime Example

```bash
# CPU inference
./onnx_inference /path/to/model.onnx

# CUDA inference (if ONNX Runtime with CUDA is installed)
./onnx_inference /path/to/model.onnx 1
```

## Example Output

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

Note: To decode tokens to text, you need to implement
      a tokenizer in C++ or use the Python tokenizer.
```

## Tokenization

The C++ examples work with token IDs directly. To convert text to tokens and back:

### Option 1: Use Python for Tokenization

Create a simple Python script to tokenize your input:

```python
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init

device_type = "cpu"
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("sft", device, phase="eval")

# Tokenize
text = "The chemical formula of water is"
bos = tokenizer.get_bos_token_id()
tokens = tokenizer.encode(text, prepend=bos)
print("Token IDs:", tokens)

# Detokenize
generated_tokens = [1, 464, 11742, 15150, 315, 3090, 374, 473]
text = tokenizer.decode(generated_tokens)
print("Text:", text)
```

### Option 2: Implement Tokenizer in C++

You can implement a BPE tokenizer in C++ using the vocabulary file from the trained model. The nanochat tokenizer is compatible with tiktoken format.

## Performance Tips

1. **Use CUDA**: If you have a GPU, use CUDA for much faster inference
2. **Batch Processing**: Modify the examples to process multiple sequences in parallel
3. **KV Cache**: For production use, implement KV caching to avoid recomputing past tokens
4. **Quantization**: Consider quantizing the model for faster inference and lower memory usage

## Limitations

The exported models have some limitations compared to the Python version:

1. **No Tool Use**: Calculator and other tool features are not included in the exported model
2. **No Special Token Handling**: Special tokens like `<|python_start|>` are not automatically handled
3. **Simplified Generation**: The examples use basic sampling; you may want to implement more sophisticated decoding strategies

## Troubleshooting

### LibTorch Issues

- **Error: "libtorch not found"**: Make sure `CMAKE_PREFIX_PATH` points to the LibTorch directory
- **Runtime errors**: Ensure the LibTorch version matches the PyTorch version used for export
- **CUDA errors**: Verify CUDA versions match between LibTorch and your system

### ONNX Runtime Issues

- **Error: "onnxruntime not found"**: Set `ONNXRUNTIME_DIR` environment variable
- **Model loading fails**: Ensure the ONNX model was exported successfully
- **Numerical differences**: Small differences (<1e-3) are normal due to floating-point precision

### General Issues

- **Out of memory**: Reduce batch size or sequence length
- **Slow inference**: Use GPU acceleration or consider model quantization
- **Wrong outputs**: Verify the exported model produces correct outputs in Python first

## Further Reading

- [LibTorch Documentation](https://pytorch.org/cppdocs/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [nanochat Export Documentation](../../README.md#model-export)

## License

MIT License - see the main repository LICENSE file.
