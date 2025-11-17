#!/usr/bin/env python3
"""
Export nanochat models to TorchScript and ONNX formats.

This script exports trained nanochat models to formats that can be used
for inference in C++, C#, Java, and other languages.

Supported formats:
- TorchScript (.pt): For use with LibTorch (C++ PyTorch API)
- ONNX (.onnx): For use with ONNX Runtime (cross-platform)

Usage examples:

# Export SFT model to TorchScript
python -m scripts.export_model --source sft --format torchscript --output model.pt

# Export to ONNX
python -m scripts.export_model --source sft --format onnx --output model.onnx

# Export with specific model tag and step
python -m scripts.export_model --source mid --model-tag d20 --step 10000 --format both

# Export with KV cache support (experimental)
python -m scripts.export_model --source sft --format torchscript --with-cache --output model_cache.pt
"""

import argparse
import os
import torch
import torch.nn as nn
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.export_wrapper import ExportableGPT, ExportableGPTWithCache


def export_to_torchscript(
    model,
    output_path: str,
    with_cache: bool = False,
    max_seq_len: int = 4096,
    example_seq_len: int = 32,
    device: torch.device = None
):
    """
    Export model to TorchScript format.
    
    Args:
        model: The GPT model to export
        output_path: Path to save the exported model
        with_cache: Whether to include KV cache support
        max_seq_len: Maximum sequence length for rotary embeddings
        example_seq_len: Sequence length for tracing
        device: Device to use for export
    """
    print(f"Exporting to TorchScript (with_cache={with_cache})...")
    
    # Create wrapper
    if with_cache:
        wrapper = ExportableGPTWithCache(model, max_seq_len=max_seq_len)
    else:
        wrapper = ExportableGPT(model, max_seq_len=max_seq_len)
    
    wrapper.eval()
    
    # Create example inputs for tracing
    batch_size = 1
    example_input_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, example_seq_len),
        dtype=torch.long,
        device=device
    )
    
    if with_cache:
        # Example with cache
        n_layers = model.config.n_layer
        n_kv_head = model.config.n_kv_head
        head_dim = model.config.n_embd // model.config.n_head
        
        example_cache_k = torch.zeros(
            n_layers, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        example_cache_v = torch.zeros(
            n_layers, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        example_position = torch.tensor(0, dtype=torch.long, device=device)
        
        example_inputs = (example_input_ids, example_cache_k, example_cache_v, example_position)
    else:
        example_inputs = (example_input_ids,)
    
    # Trace the model
    print("Tracing model with example inputs...")
    try:
        traced_model = torch.jit.trace(wrapper, example_inputs)
        
        # Save the traced model
        print(f"Saving TorchScript model to {output_path}...")
        traced_model.save(output_path)
        print(f"✓ Successfully exported to TorchScript: {output_path}")
        
        # Verify the export
        print("Verifying export...")
        with torch.no_grad():
            original_output = wrapper(*example_inputs)
            loaded_model = torch.jit.load(output_path)
            traced_output = loaded_model(*example_inputs)
            
            if with_cache:
                # Compare logits only (first output)
                max_diff = torch.max(torch.abs(original_output[0] - traced_output[0])).item()
            else:
                max_diff = torch.max(torch.abs(original_output - traced_output)).item()
            
            print(f"  Max difference between original and traced: {max_diff:.6e}")
            if max_diff < 1e-4:
                print("  ✓ Verification passed!")
            else:
                print(f"  ⚠ Warning: Difference is larger than expected")
        
        return True
    except Exception as e:
        print(f"✗ Failed to export to TorchScript: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_to_onnx(
    model,
    output_path: str,
    with_cache: bool = False,
    max_seq_len: int = 4096,
    example_seq_len: int = 32,
    device: torch.device = None,
    opset_version: int = 17
):
    """
    Export model to ONNX format.
    
    Args:
        model: The GPT model to export
        output_path: Path to save the exported model
        with_cache: Whether to include KV cache support
        max_seq_len: Maximum sequence length for rotary embeddings
        example_seq_len: Sequence length for export
        device: Device to use for export
        opset_version: ONNX opset version
    """
    print(f"Exporting to ONNX (with_cache={with_cache}, opset={opset_version})...")
    
    # Create wrapper
    if with_cache:
        wrapper = ExportableGPTWithCache(model, max_seq_len=max_seq_len)
    else:
        wrapper = ExportableGPT(model, max_seq_len=max_seq_len)
    
    wrapper.eval()
    
    # Create example inputs
    batch_size = 1
    example_input_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, example_seq_len),
        dtype=torch.long,
        device=device
    )
    
    if with_cache:
        n_layers = model.config.n_layer
        n_kv_head = model.config.n_kv_head
        head_dim = model.config.n_embd // model.config.n_head
        
        example_cache_k = torch.zeros(
            n_layers, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        example_cache_v = torch.zeros(
            n_layers, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=torch.bfloat16, device=device
        )
        example_position = torch.tensor(0, dtype=torch.long, device=device)
        
        example_inputs = (example_input_ids, example_cache_k, example_cache_v, example_position)
        input_names = ["input_ids", "cache_k", "cache_v", "position"]
        output_names = ["logits", "cache_k_out", "cache_v_out"]
        
        # Dynamic axes for variable sequence length and batch size
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
    else:
        example_inputs = (example_input_ids,)
        input_names = ["input_ids"]
        output_names = ["logits"]
        
        # Dynamic axes for variable sequence length and batch size
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
    
    # Export to ONNX
    print("Exporting model to ONNX format...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example_inputs,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )
        
        print(f"✓ Successfully exported to ONNX: {output_path}")
        
        # Verify with ONNX
        try:
            import onnx
            print("Verifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("  ✓ ONNX model is valid!")
            
            # Try to verify with ONNX Runtime if available
            try:
                import onnxruntime as ort
                print("Verifying with ONNX Runtime...")
                
                # Create inference session
                ort_session = ort.InferenceSession(
                    output_path,
                    providers=['CPUExecutionProvider']
                )
                
                # Prepare inputs
                if with_cache:
                    ort_inputs = {
                        "input_ids": example_input_ids.cpu().numpy(),
                        "cache_k": example_cache_k.cpu().numpy(),
                        "cache_v": example_cache_v.cpu().numpy(),
                        "position": example_position.cpu().numpy(),
                    }
                else:
                    ort_inputs = {
                        "input_ids": example_input_ids.cpu().numpy(),
                    }
                
                # Run inference
                ort_outputs = ort_session.run(None, ort_inputs)
                
                # Compare with PyTorch
                with torch.no_grad():
                    torch_outputs = wrapper(*example_inputs)
                    if with_cache:
                        torch_logits = torch_outputs[0].cpu().numpy()
                    else:
                        torch_logits = torch_outputs.cpu().numpy()
                    
                    ort_logits = ort_outputs[0]
                    max_diff = abs(torch_logits - ort_logits).max()
                    
                    print(f"  Max difference between PyTorch and ONNX Runtime: {max_diff:.6e}")
                    if max_diff < 1e-3:
                        print("  ✓ ONNX Runtime verification passed!")
                    else:
                        print(f"  ⚠ Warning: Difference is larger than expected")
                
            except ImportError:
                print("  ⓘ ONNX Runtime not available, skipping runtime verification")
        
        except ImportError:
            print("  ⓘ ONNX package not available, skipping validation")
        
        return True
    
    except Exception as e:
        print(f"✗ Failed to export to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export nanochat models to TorchScript and ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="sft",
        choices=["base", "mid", "sft", "rl"],
        help="Model source to export (default: sft)"
    )
    parser.add_argument(
        "--model-tag", "-g",
        type=str,
        default=None,
        help="Model tag to load (e.g., d20, d26)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific checkpoint step to load"
    )
    
    # Export options
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="torchscript",
        choices=["torchscript", "onnx", "both"],
        help="Export format (default: torchscript)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: model.pt or model.onnx)"
    )
    parser.add_argument(
        "--with-cache",
        action="store_true",
        help="Export with KV cache support (experimental, may not work with ONNX)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length for rotary embeddings (default: 4096)"
    )
    parser.add_argument(
        "--example-seq-len",
        type=int,
        default=32,
        help="Sequence length for tracing/export (default: 32)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    
    # Device options
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps", ""],
        help="Device type: cuda|cpu|mps (empty = autodetect)"
    )
    
    args = parser.parse_args()
    
    # Initialize device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    print("="*60)
    print("nanochat Model Export")
    print("="*60)
    print(f"Source: {args.source}")
    print(f"Model tag: {args.model_tag or 'default'}")
    print(f"Step: {args.step or 'latest'}")
    print(f"Format: {args.format}")
    print(f"Device: {device}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"With KV cache: {args.with_cache}")
    print("="*60)
    
    # Load the model
    print("\nLoading model...")
    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step
    )
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Config: {model.config}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Determine output paths
    if args.output:
        base_path = args.output.rsplit(".", 1)[0]
        ext = args.output.rsplit(".", 1)[1] if "." in args.output else ""
    else:
        base_path = f"nanochat_{args.source}"
        if args.model_tag:
            base_path += f"_{args.model_tag}"
        if args.step:
            base_path += f"_step{args.step}"
        if args.with_cache:
            base_path += "_cache"
        ext = ""
    
    # Export to requested formats
    success = True
    
    if args.format in ["torchscript", "both"]:
        output_path = f"{base_path}.pt" if not ext or ext == "pt" else args.output
        success &= export_to_torchscript(
            model,
            output_path,
            with_cache=args.with_cache,
            max_seq_len=args.max_seq_len,
            example_seq_len=args.example_seq_len,
            device=device
        )
    
    if args.format in ["onnx", "both"]:
        output_path = f"{base_path}.onnx" if not ext or ext == "onnx" else args.output
        success &= export_to_onnx(
            model,
            output_path,
            with_cache=args.with_cache,
            max_seq_len=args.max_seq_len,
            example_seq_len=args.example_seq_len,
            device=device,
            opset_version=args.opset_version
        )
    
    print("\n" + "="*60)
    if success:
        print("✓ Export completed successfully!")
        print("\nNext steps:")
        print("  - For TorchScript: Use LibTorch C++ API to load and run inference")
        print("  - For ONNX: Use ONNX Runtime in C++, C#, Java, or other languages")
        print("\nSee examples/cpp_inference/ for C++ usage examples")
    else:
        print("✗ Export failed. See errors above.")
    print("="*60)


if __name__ == "__main__":
    main()
