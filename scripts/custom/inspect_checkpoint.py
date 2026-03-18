import torch
import json
import os

# Path to the checkpoints
CHECKPOINT_DIR = os.path.expanduser("~/.cache/nanochat/base_checkpoints/d2/")
STEP = "000100"

model_path = os.path.join(CHECKPOINT_DIR, f"model_{STEP}.pt")
meta_path = os.path.join(CHECKPOINT_DIR, f"meta_{STEP}.json")
optim_path = os.path.join(CHECKPOINT_DIR, f"optim_{STEP}_rank0.pt")

print(f"--- Inspecting Checkpoint Step {STEP} ---\n")

# 1. Inspect Metadata
if os.path.exists(meta_path):
    print(f"Loading Metadata: {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        print(json.dumps(meta, indent=2))
else:
    print(f"Metadata not found at {meta_path}")

print("\n" + "="*50 + "\n")

# 2. Inspect Model Parameters
if os.path.exists(model_path):
    print(f"Loading Model Parameters: {model_path}")
    # Load on CPU to avoid MPS/CUDA issues during inspection
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    print(f"Number of parameter tensors: {len(state_dict)}")
    print(f"{'Layer Name':<40} | {'Shape':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 88)
    
    for name, param in list(state_dict.items())[:20]: # Show first 20 layers
        print(f"{name:<40} | {str(list(param.shape)):<20} | {param.float().mean():.4f} | {param.float().std():.4f}")
    
    if len(state_dict) > 20:
        print(f"... and {len(state_dict) - 20} more layers.")
else:
    print(f"Model file not found at {model_path}")

print("\n" + "="*50 + "\n")

# 3. Inspect Optimizer State (Summary)
if os.path.exists(optim_path):
    print(f"Loading Optimizer State: {optim_path}")
    optim_state = torch.load(optim_path, map_location='cpu')
    print(f"Keys in optimizer state: {list(optim_state.keys())}")
    if 'state' in optim_state:
        print(f"Number of optimized parameters: {len(optim_state['state'])}")
else:
    print(f"Optimizer file not found at {optim_path}")
