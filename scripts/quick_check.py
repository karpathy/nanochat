"""Quick checkpoint structure check."""
import torch
import sys

checkpoint_path = "/raid/diana/nanochat_cache/chatsft_checkpoints/d20/model_000650.pt"
print(f"Loading: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\n" + "="*60)
    print("CHECKPOINT STRUCTURE")
    print("="*60)

    print(f"\nTop-level keys: {list(checkpoint.keys())}\n")

    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"'{key}': dict with {len(value)} items")
            # Show a few sub-keys if it's a dict
            sub_keys = list(value.keys())[:3]
            print(f"  Sample keys: {sub_keys}")
        elif isinstance(value, torch.Tensor):
            print(f"'{key}': Tensor {value.shape}, dtype={value.dtype}")
        else:
            print(f"'{key}': {type(value).__name__} = {value}")

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    # Check what we need
    has_model = 'model' in checkpoint
    has_config = 'config' in checkpoint
    has_state_dict = 'state_dict' in checkpoint
    has_model_state_dict = 'model_state_dict' in checkpoint

    print(f"\n✓ Has 'model' key: {has_model}")
    print(f"✓ Has 'config' key: {has_config}")
    print(f"✓ Has 'state_dict' key: {has_state_dict}")
    print(f"✓ Has 'model_state_dict' key: {has_model_state_dict}")

    # Try to infer the structure
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    if has_model and has_config:
        print("\n✅ Checkpoint has expected structure!")
        print("   No changes needed to benchmark_optimizations.py")
    elif has_state_dict:
        print("\n⚠️  Checkpoint uses 'state_dict' instead of 'model'")
        print("   Need to update benchmark to use checkpoint['state_dict']")
    elif has_model_state_dict:
        print("\n⚠️  Checkpoint uses 'model_state_dict' instead of 'model'")
        print("   Need to update benchmark to use checkpoint['model_state_dict']")
    else:
        print("\n❌ Checkpoint has unexpected structure!")
        print("   Available keys:", list(checkpoint.keys()))
        print("   You may need to check how the model was saved during training")

except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
