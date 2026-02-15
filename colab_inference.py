#!/usr/bin/env python3
"""
colab_inference.py - Run nanochat inference in Google Colab

Instructions for Google Colab:
1. Upload your converted model directory (from convert_to_hf.py) to Colab files
2. Install dependencies: !pip install torch transformers tokenizers
3. Update MODEL_DIR to match your uploaded folder name
4. Run the cells below

Example usage:
    model = NanoChatInference("nanochat-d20-hf")
    model.generate("The meaning of life is", max_tokens=50)
"""

import torch
import json
import sys
from pathlib import Path

# For Colab: uncomment this line
# !pip install torch transformers tokenizers

class NanoChatInference:
    """Simple inference wrapper for nanochat models"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔥 Using device: {self.device}")
        
        # Load configuration
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load model weights
        model_path = Path(model_dir) / "pytorch_model.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        self.state_dict = torch.load(model_path, map_location=self.device)
        
        # Load metadata if available
        metadata_path = Path(model_dir) / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            print(f"📊 Model trained to step: {self.metadata.get('step', 'unknown')}")
            print(f"📉 Final loss: {self.metadata.get('loss', 'unknown')}")
        
        print(f"✅ Loaded model from: {model_dir}")
        print(f"📐 Architecture: {self.config['num_hidden_layers']} layers, {self.config['hidden_size']} hidden size")
        print(f"📝 Vocab size: {self.config['vocab_size']}")
        print(f"💾 Model parameters: {len(self.state_dict)} tensors")
        
        # Show some parameter shapes for debugging
        print("🔍 Sample parameter shapes:")
        for i, (key, tensor) in enumerate(self.state_dict.items()):
            if i < 5:  # Show first 5
                print(f"   {key}: {tensor.shape}")
    
    def show_model_info(self):
        """Display detailed model information"""
        print(f"\n📋 Model Information:")
        print(f"   Directory: {self.model_dir}")
        print(f"   Device: {self.device}")
        print(f"   Layers: {self.config['num_hidden_layers']}")
        print(f"   Hidden size: {self.config['hidden_size']}")
        print(f"   Attention heads: {self.config['num_attention_heads']}")
        print(f"   Vocab size: {self.config['vocab_size']}")
        print(f"   Max sequence length: {self.config['max_position_embeddings']}")
        
        if hasattr(self, 'metadata'):
            print(f"\n📊 Training Info:")
            print(f"   Step: {self.metadata.get('step', 'unknown')}")
            print(f"   Loss: {self.metadata.get('loss', 'unknown')}")
    
    def generate_placeholder(self, prompt, max_tokens=100, temperature=0.8):
        """
        Placeholder generation function
        
        Note: This is a demo function. For actual generation, you need to:
        1. Implement the nanochat model architecture in PyTorch
        2. Load the state_dict into the model
        3. Implement tokenization (nanochat uses GPT tokenizer)
        4. Implement the generation loop
        """
        print(f"\n🎯 Generation Request:")
        print(f"   Prompt: '{prompt}'")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        
        print(f"\n⚠️  This is a placeholder! To enable actual generation:")
        print(f"   1. Import nanochat model classes")
        print(f"   2. Load state_dict into model")
        print(f"   3. Implement tokenization")
        print(f"   4. Run generation loop")
        
        # Show that model weights are loaded
        print(f"\n✅ Model weights are ready:")
        total_params = sum(p.numel() for p in self.state_dict.values())
        print(f"   Total parameters: {total_params:,}")
        
        return f"[Generated text would appear here - model has {total_params:,} parameters ready]"

# Example usage for Colab
if __name__ == "__main__":
    # Update this to match your uploaded model directory
    MODEL_DIR = "nanochat-d20-hf"  # Change this to your uploaded folder name
    
    print("🚀 nanochat Colab Inference Demo")
    print("=" * 50)
    
    try:
        # Initialize model
        model = NanoChatInference(MODEL_DIR)
        
        # Show model info
        model.show_model_info()
        
        # Demo generation (placeholder)
        result = model.generate_placeholder("The meaning of life is", max_tokens=50)
        print(f"\n📝 Result: {result}")
        
        print(f"\n🎉 Model loaded successfully! Ready for implementation.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure you:")
        print(f"   1. Uploaded your model directory to Colab files")
        print(f"   2. Updated MODEL_DIR to match the folder name")
        print(f"   3. Ran convert_to_hf.py first to create the HF format")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"   Check your model files and try again")