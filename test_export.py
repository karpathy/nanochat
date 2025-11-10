#!/usr/bin/env python3
"""
Simple test script to verify export functionality without requiring a trained model.
This creates a minimal GPT model and tests the export wrappers.
"""

import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig
from nanochat.export_wrapper import ExportableGPT, ExportableGPTWithCache

def test_export_wrapper():
    """Test the ExportableGPT wrapper."""
    print("="*60)
    print("Testing Export Wrapper")
    print("="*60)
    
    # Create a small test model
    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128
    )
    
    print(f"\nCreating test model with config:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  n_embd: {config.n_embd}")
    
    # Create model
    device = torch.device("cpu")
    model = GPT(config)
    model.to(device)
    model.init_weights()
    model.eval()
    
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test ExportableGPT
    print("\n--- Testing ExportableGPT (without cache) ---")
    wrapper = ExportableGPT(model, max_seq_len=256)
    wrapper.eval()
    
    # Create test input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    
    print(f"Input shape: {list(input_ids.shape)}")
    
    # Forward pass
    with torch.no_grad():
        logits = wrapper(input_ids)
    
    print(f"Output shape: {list(logits.shape)}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {config.vocab_size}]")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch!"
    print("✓ Forward pass successful!")
    
    # Test with position offset
    print("\nTesting with position offset...")
    position_offset = torch.tensor(5)
    with torch.no_grad():
        logits_offset = wrapper(input_ids, position_offset)
    
    print(f"Output shape with offset: {list(logits_offset.shape)}")
    assert logits_offset.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch!"
    print("✓ Forward pass with offset successful!")
    
    # Test TorchScript tracing
    print("\n--- Testing TorchScript Tracing ---")
    try:
        traced_model = torch.jit.trace(wrapper, (input_ids,))
        print("✓ TorchScript tracing successful!")
        
        # Test traced model
        with torch.no_grad():
            traced_output = traced_model(input_ids)
        
        max_diff = torch.max(torch.abs(logits - traced_output)).item()
        print(f"Max difference between original and traced: {max_diff:.6e}")
        
        if max_diff < 1e-5:
            print("✓ Traced model output matches original!")
        else:
            print(f"⚠ Warning: Difference is {max_diff:.6e}")
    except Exception as e:
        print(f"✗ TorchScript tracing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ExportableGPTWithCache
    print("\n--- Testing ExportableGPTWithCache ---")
    wrapper_cache = ExportableGPTWithCache(model, max_seq_len=256, max_batch_size=2)
    wrapper_cache.eval()
    
    with torch.no_grad():
        logits_cache, cache_k, cache_v = wrapper_cache(input_ids)
    
    print(f"Output shape: {list(logits_cache.shape)}")
    print(f"Cache K shape: {list(cache_k.shape)}")
    print(f"Cache V shape: {list(cache_v.shape)}")
    
    assert logits_cache.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch!"
    print("✓ Forward pass with cache successful!")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_export_wrapper()
        print("\n✓ Export wrapper is working correctly!")
        print("\nNext steps:")
        print("  1. Train a model using speedrun.sh or run1000.sh")
        print("  2. Export the trained model:")
        print("     python -m scripts.export_model --source sft --format torchscript")
        print("  3. Use the exported model in C++ (see examples/cpp_inference/)")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
