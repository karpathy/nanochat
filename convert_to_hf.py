#!/usr/bin/env python3
"""
convert_to_hf.py - Convert nanochat checkpoint to Hugging Face format

Usage:
    python convert_to_hf.py <checkpoint.pt> <output_dir>
    
Example:
    python convert_to_hf.py checkpoints/nanochat-d20/step_2500.pt nanochat-d20-hf
"""

import torch
import json
import sys
from pathlib import Path

def convert_nanochat_to_hf(checkpoint_path, output_dir):
    """Convert nanochat checkpoint to HF-compatible format"""
    
    print(f"🔄 Converting {checkpoint_path} to Hugging Face format...")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
    
    # Extract model state and config
    model_state = checkpoint['model']
    config = checkpoint.get('config', {})
    
    print(f"📊 Model has {len(model_state)} parameter tensors")
    print(f"⚙️  Config keys: {list(config.keys())}")
    
    # Create HF-style config.json
    hf_config = {
        "model_type": "nanochat",
        "vocab_size": config.get('vocab_size', 50304),
        "hidden_size": config.get('n_embd', 768),
        "num_attention_heads": config.get('n_head', 12), 
        "num_hidden_layers": config.get('n_layer', 12),
        "intermediate_size": config.get('n_embd', 768) * 4,
        "max_position_embeddings": config.get('block_size', 1024),
        "layer_norm_eps": 1e-5,
        "use_cache": True,
        "torch_dtype": "float32",
        "architectures": ["NanoChatLMHeadModel"],
        "_nanochat_original_config": config  # Preserve original for reference
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model_state, output_path / "pytorch_model.bin")
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    # Save training metadata if available
    metadata = {
        "step": checkpoint.get('step', 'unknown'),
        "loss": checkpoint.get('val_loss', 'unknown'),
        "conversion_info": {
            "source_checkpoint": str(checkpoint_path),
            "converted_by": "nanochat convert_to_hf.py",
            "model_architecture": "nanochat GPT-style transformer"
        }
    }
    
    with open(output_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Conversion complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files created:")
    print(f"   - pytorch_model.bin (model weights)")
    print(f"   - config.json (HF configuration)")
    print(f"   - training_metadata.json (training info)")
    
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_to_hf.py <checkpoint.pt> <output_dir>")
        print("\nExample:")
        print("  python convert_to_hf.py checkpoints/nanochat-d20/step_2500.pt nanochat-d20-hf")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    success = convert_nanochat_to_hf(checkpoint_path, output_dir)
    
    if success:
        print(f"\n🚀 Ready for inference! Upload '{output_dir}' to Google Colab or use locally.")
    else:
        print("❌ Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()