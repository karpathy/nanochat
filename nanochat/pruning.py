"""
Magnitude-based pruning for nanochat.
Zero out small weights to reduce model density.
"""

import torch
import torch.nn as nn
from nanochat.common import print0

def prune_model(model, amount=0.3):
    """
    Apply magnitude-based pruning globally to all Linear layers in the model.

    Args:
        model: The model to prune.
        amount: Fraction of weights to zero out (0.3 = prune 30%).
    """
    # Collect all weights from Linear layers
    all_weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            all_weights.append(module.weight.data.flatten())

    if not all_weights:
        print0("No Linear layers found to prune.")
        return

    # Find the global threshold for pruning
    # Move to CPU for threshold calculation to avoid OOM on GPU
    all_weights_flat = torch.cat(all_weights).cpu()
    
    # torch.quantile has a limit on input size (2^31). For large models, we subsample.
    if all_weights_flat.numel() > 10_000_000:
        # Subsample 10M elements for threshold calculation (plenty for accuracy)
        indices = torch.randperm(all_weights_flat.numel())[:10_000_000]
        sample = all_weights_flat[indices]
        threshold = torch.quantile(sample.abs(), amount).to(model.get_device())
    else:
        threshold = torch.quantile(all_weights_flat.abs(), amount).to(model.get_device())

    total_params = 0
    pruned_params = 0

    # Apply the mask
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            mask = weight.abs() > threshold
            weight.mul_(mask.to(weight.dtype))

            total_params += weight.numel()
            pruned_params += (mask == 0).sum().item()

    print0(f"Pruned {pruned_params} parameters out of {total_params} total ({100 * pruned_params / total_params:.2f}%)")
    return threshold
