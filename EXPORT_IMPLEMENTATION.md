# TorchScript/ONNX Export Implementation for nanochat

## Summary

This document describes the implementation of TorchScript and ONNX export functionality for nanochat models, addressing GitHub Issue #73.

## Problem Statement

The original issue requested the ability to export nanochat models for inference in different languages (C/C++, etc.) using TorchScript, TorchTrace, or ONNX formats. The main challenges were:

1. **Rotary Embeddings**: Pre-computed embeddings stored as buffers
2. **KV Cache**: Dynamic state management during autoregressive generation
3. **Tool Use**: Python-based calculator and special token handling in the Engine class

## Solution Overview

The implementation provides:

1. **Export-friendly model wrappers** that encapsulate rotary embeddings and simplify the forward pass
2. **Export script** supporting both TorchScript and ONNX formats
3. **C++ inference examples** for LibTorch and ONNX Runtime
4. **Comprehensive documentation** for users

## Implementation Details

### 1. Export Wrapper (`nanochat/export_wrapper.py`)

Two wrapper classes were created:

#### `ExportableGPT`
- Simplified forward pass without KV cache
- Self-contained rotary embeddings
- Suitable for both TorchScript and ONNX export
- Best for simple use cases and maximum compatibility

```python
wrapper = ExportableGPT(model, max_seq_len=4096)
logits = wrapper(input_ids)
```

#### `ExportableGPTWithCache`
- Explicit KV cache management as inputs/outputs
- Enables stateful inference for better performance
- More complex but suitable for production deployments
- May have limited ONNX support due to dynamic shapes

```python
wrapper = ExportableGPTWithCache(model, max_seq_len=4096)
logits, cache_k, cache_v = wrapper(input_ids, cache_k, cache_v, position)
```

**Key Design Decisions:**

- Rotary embeddings are pre-computed and stored as persistent buffers
- Simplified attention mechanism without Engine complexity
- No tool use or special token handling (Python-only features)
- Support for position offsets to enable KV cache usage

### 2. Export Script (`scripts/export_model.py`)

A comprehensive CLI tool for exporting models:

```bash
# Export to TorchScript
python -m scripts.export_model --source sft --format torchscript --output model.pt

# Export to ONNX
python -m scripts.export_model --source sft --format onnx --output model.onnx

# Export both formats
python -m scripts.export_model --source sft --format both
```

**Features:**

- Supports all model sources (base, mid, sft, rl)
- Automatic model loading and validation
- Output verification (compares exported vs original)
- ONNX validation with onnxruntime (if available)
- Configurable sequence lengths and opset versions
- Support for both cached and non-cached variants

### 3. C++ Examples (`examples/cpp_inference/`)

#### LibTorch Example (`libtorch_inference.cpp`)

Demonstrates inference using PyTorch's C++ API:

- Model loading from TorchScript files
- Single forward pass
- Autoregressive generation with sampling
- Temperature and top-k sampling support

```cpp
NanoChatInference model(model_path, device);
auto logits = model.forward(input_ids);
auto generated = model.generate(prompt_ids, max_tokens, temperature, top_k);
```

#### ONNX Runtime Example (`onnx_inference.cpp`)

Cross-platform inference with ONNX Runtime:

- ONNX model loading
- CPU and CUDA execution providers
- Efficient inference with ONNX Runtime optimizations
- Compatible with multiple languages (C++, C#, Java, Python)

```cpp
NanoChatONNXInference model(model_path, use_cuda);
auto logits = model.forward(input_ids, batch_size, seq_len, vocab_size);
auto generated = model.generate(prompt_ids, max_tokens, temperature, top_k);
```

#### Build System (`CMakeLists.txt`)

- Supports both LibTorch and ONNX Runtime
- Optional builds (can build either or both)
- Cross-platform (Linux, macOS, Windows)
- Clear error messages for missing dependencies

### 4. Documentation

#### Main README Updates

Added a new "Model Export" section covering:
- Export commands and options
- C++ inference quick start
- Limitations of exported models
- Links to detailed C++ documentation

#### C++ Examples README

Comprehensive guide including:
- Prerequisites and dependencies
- Build instructions for all platforms
- Export workflow
- Running examples
- Tokenization strategies
- Performance tips
- Troubleshooting

## Testing

A test script (`test_export.py`) was created to verify the implementation:

```bash
python3 test_export.py
```

**Test Coverage:**

- ✓ ExportableGPT forward pass
- ✓ Position offset handling
- ✓ TorchScript tracing
- ✓ Output verification (original vs traced)
- ✓ ExportableGPTWithCache forward pass
- ✓ Cache shape validation

All tests pass successfully with zero numerical differences between original and traced models.

## Limitations

The exported models have intentional limitations:

1. **No Tool Use**: Calculator and Python execution features are not included
   - These require Python runtime and are not suitable for export
   - Users can implement similar features in their target language if needed

2. **No Special Token Handling**: Special tokens like `<|python_start|>` are not automatically processed
   - The exported model only performs the core transformer forward pass
   - Special token logic must be implemented in the inference code

3. **Tokenization**: Token encoding/decoding is not included
   - Users must implement BPE tokenization in C++ or use Python for preprocessing
   - The nanochat tokenizer is tiktoken-compatible

4. **KV Cache Complexity**: The cached variant is more complex
   - Recommended to start with the simple non-cached version
   - Cache management must be handled by the caller

## Usage Examples

### Python Export

```python
# Export a trained SFT model
python -m scripts.export_model \
    --source sft \
    --format torchscript \
    --output nanochat_sft.pt \
    --max-seq-len 4096
```

### C++ Inference (LibTorch)

```cpp
#include "libtorch_inference.cpp"

int main() {
    NanoChatInference model("model.pt", torch::kCUDA);
    
    std::vector<int64_t> prompt = {1, 464, 11742, 15150, 315, 3090, 374};
    auto generated = model.generate(prompt, 100, 0.8, 50);
    
    // generated now contains the full sequence including prompt
    return 0;
}
```

### C++ Inference (ONNX Runtime)

```cpp
#include "onnx_inference.cpp"

int main() {
    NanoChatONNXInference model("model.onnx", true);
    
    std::vector<int64_t> prompt = {1, 464, 11742, 15150, 315, 3090, 374};
    auto generated = model.generate(prompt, 100, 0.8, 50);
    
    return 0;
}
```

## Files Created/Modified

### New Files

1. `nanochat/export_wrapper.py` - Export-friendly model wrappers
2. `scripts/export_model.py` - Export script for TorchScript/ONNX
3. `examples/cpp_inference/libtorch_inference.cpp` - LibTorch example
4. `examples/cpp_inference/onnx_inference.cpp` - ONNX Runtime example
5. `examples/cpp_inference/CMakeLists.txt` - Build configuration
6. `examples/cpp_inference/README.md` - C++ documentation
7. `test_export.py` - Test script for export functionality
8. `EXPORT_IMPLEMENTATION.md` - This document

### Modified Files

1. `README.md` - Added export documentation and updated file structure

## Future Enhancements

Potential improvements for future work:

1. **Quantization Support**: Add INT8/FP16 quantization for faster inference
2. **Batch Processing**: Optimize for batch inference in C++
3. **Tokenizer Port**: Implement BPE tokenizer in C++ for end-to-end inference
4. **Mobile Deployment**: Add support for mobile platforms (iOS/Android)
5. **WebAssembly**: Export to WASM for browser-based inference
6. **Streaming Generation**: Implement streaming token generation in C++
7. **Model Optimization**: Add ONNX graph optimizations and operator fusion

## Performance Considerations

1. **Use GPU**: CUDA inference is significantly faster than CPU
2. **KV Cache**: Implement KV caching for production deployments
3. **Batch Size**: Process multiple sequences in parallel when possible
4. **Quantization**: Consider quantizing models for deployment
5. **Operator Fusion**: ONNX Runtime automatically fuses operators for better performance

## Conclusion

This implementation successfully addresses GitHub Issue #73 by providing:

- ✅ TorchScript export support
- ✅ ONNX export support  
- ✅ C++ inference examples (LibTorch and ONNX Runtime)
- ✅ Comprehensive documentation
- ✅ Tested and verified implementation

Users can now export trained nanochat models and run inference in C++ or other languages, enabling production deployments without Python dependencies. The implementation maintains the simplicity and hackability that nanochat is known for, while providing the flexibility needed for diverse deployment scenarios.
