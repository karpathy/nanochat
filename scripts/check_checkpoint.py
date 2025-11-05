"""Quick script to inspect checkpoint structure."""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint.py <path_to_checkpoint>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n📦 Checkpoint keys:")
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, dict):
        print(f"  - {key}: dict with {len(value)} items")
    elif isinstance(value, torch.Tensor):
        print(f"  - {key}: Tensor {value.shape}")
    else:
        print(f"  - {key}: {type(value).__name__}")

# Check for model weights
if 'model' in checkpoint:
    print("\n✅ Model weights found")
    model_dict = checkpoint['model']
    print(f"   Number of parameters: {len(model_dict)}")
    print(f"   Sample keys: {list(model_dict.keys())[:5]}")
else:
    print("\n⚠️  No 'model' key found!")
    print("   Available keys:", list(checkpoint.keys()))

# Check for config
if 'config' in checkpoint:
    print("\n✅ Config found")
    config = checkpoint['config']
    print(f"   Type: {type(config)}")
    if hasattr(config, '__dict__'):
        print(f"   Attributes: {list(vars(config).keys())[:10]}")
else:
    print("\n⚠️  No 'config' key found!")

print("\n" + "="*60)
print("RECOMMENDATION:")
if 'model' not in checkpoint or 'config' not in checkpoint:
    print("Your checkpoint is missing required keys.")
    print("Please check how the model was saved during training.")
    print("\nExpected checkpoint structure:")
    print("  checkpoint = {")
    print("      'model': model.state_dict(),")
    print("      'config': model.config,")
    print("      'optimizer': optimizer.state_dict(),  # optional")
    print("      'step': current_step,  # optional")
    print("  }")
else:
    print("✅ Checkpoint looks good!")
