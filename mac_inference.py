#!/usr/bin/env python3
"""
mac_inference.py - Run nanochat inference on Mac (with MPS GPU support)

Usage:
    python mac_inference.py <model_dir> [prompt]
    
Example:
    python mac_inference.py nanochat-d20-hf "The future of AI is"
"""

import torch
import json
import sys
from pathlib import Path

def check_mac_gpu():
    """Check Mac GPU (MPS) availability"""
    if torch.backends.mps.is_available():
        print("✅ Mac GPU (MPS) is available")
        return torch.device('mps')
    else:
        print("⚠️  Mac GPU (MPS) not available, using CPU")
        return torch.device('cpu')

class NanoChatMacInference:
    """Mac-optimized nanochat inference with MPS support"""
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.device = check_mac_gpu()
        
        # Load configuration
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load model weights
        model_path = self.model_dir / "pytorch_model.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        print(f"📦 Loading model weights...")
        self.state_dict = torch.load(model_path, map_location=self.device)
        
        # Load training metadata
        metadata_path = self.model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        print(f"✅ Model loaded on {self.device}")
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information"""
        print(f"\n📋 Model Information:")
        print(f"   📁 Directory: {self.model_dir}")
        print(f"   🔥 Device: {self.device}")
        print(f"   📐 Layers: {self.config['num_hidden_layers']}")
        print(f"   🧠 Hidden size: {self.config['hidden_size']}")
        print(f"   👁️  Attention heads: {self.config['num_attention_heads']}")
        print(f"   📝 Vocab size: {self.config['vocab_size']}")
        print(f"   📏 Max sequence: {self.config['max_position_embeddings']}")
        
        if self.metadata:
            step = self.metadata.get('step', 'unknown')
            loss = self.metadata.get('loss', 'unknown')
            print(f"   📊 Training step: {step}")
            print(f"   📉 Final loss: {loss}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.state_dict.values())
        print(f"   🔢 Parameters: {total_params:,}")
        
        # Check memory usage on MPS
        if self.device.type == 'mps':
            try:
                # Move a small tensor to check MPS memory
                test_tensor = torch.randn(10, 10).to(self.device)
                print(f"   ✅ MPS memory test passed")
                del test_tensor
            except Exception as e:
                print(f"   ⚠️  MPS memory test failed: {e}")
    
    def generate_with_nanochat(self, prompt, max_tokens=100, temperature=0.8):
        """
        Generate text using nanochat model
        
        Note: This requires the actual nanochat model implementation.
        You need to:
        1. Import nanochat's GPT model class
        2. Create model instance with config
        3. Load state_dict
        4. Implement generation
        """
        
        print(f"\n🎯 Generating text...")
        print(f"   📝 Prompt: '{prompt}'")
        print(f"   🎲 Max tokens: {max_tokens}")
        print(f"   🌡️  Temperature: {temperature}")
        
        try:
            # This is where you'd implement actual generation
            # Example pseudocode:
            
            # from nanochat.model import GPT, GPTConfig
            # config = GPTConfig(**self.config['_nanochat_original_config'])
            # model = GPT(config)
            # model.load_state_dict(self.state_dict)
            # model = model.to(self.device)
            # model.eval()
            
            # with torch.no_grad():
            #     encoded = tokenize(prompt)
            #     tokens = torch.tensor(encoded).unsqueeze(0).to(self.device)
            #     generated = model.generate(tokens, max_new_tokens=max_tokens, temperature=temperature)
            #     result = detokenize(generated[0])
            #     return result
            
            print(f"   ⚠️  Actual generation not implemented yet")
            print(f"   💡 Need to integrate with nanochat model classes")
            
            return f"[Placeholder: Generated {max_tokens} tokens from '{prompt}']"
            
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            return None
    
    def benchmark_speed(self):
        """Simple speed benchmark on Mac"""
        print(f"\n⏱️  Running speed benchmark...")
        
        # Test tensor operations on the target device
        size = 1000
        iterations = 10
        
        import time
        
        start_time = time.time()
        for i in range(iterations):
            x = torch.randn(size, size).to(self.device)
            y = torch.randn(size, size).to(self.device)
            z = torch.mm(x, y)
            if self.device.type == 'mps':
                torch.mps.synchronize()  # Wait for MPS completion
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        print(f"   🚀 Average matrix multiplication ({size}x{size}): {avg_time:.4f}s")
        print(f"   📊 Device performance: {'Good' if avg_time < 0.1 else 'Moderate' if avg_time < 0.5 else 'Slow'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python mac_inference.py <model_dir> [prompt]")
        print("\nExample:")
        print("  python mac_inference.py nanochat-d20-hf 'The future of AI is'")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, I am"
    
    print("🍎 nanochat Mac Inference")
    print("=" * 40)
    
    try:
        # Load model
        model = NanoChatMacInference(model_dir)
        
        # Run benchmark
        model.benchmark_speed()
        
        # Generate text (placeholder for now)
        result = model.generate_with_nanochat(prompt)
        if result:
            print(f"\n📝 Generated:")
            print(f"   {result}")
        
        print(f"\n✅ Mac inference demo complete!")
        print(f"💡 To enable actual generation, integrate with nanochat model classes")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Make sure you:")
        print(f"   1. Ran download_weights.sh to get the checkpoints")
        print(f"   2. Ran convert_to_hf.py to create the HF format")
        print(f"   3. Specified the correct model directory")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()