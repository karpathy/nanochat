"""
INT8 Weight-Only Quantization for nanochat.
Reduces model size by ~4x with minimal impact on accuracy for large models.
On-the-fly dequantization is used for compatibility across all hardware (CPU/GPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.common import COMPUTE_DTYPE

class Int8WeightOnlyLinear(nn.Module):
    """
    Linear layer that stores weights as INT8 and dequantizes them to COMPUTE_DTYPE in forward.
    This reduces VRAM usage for weights but does compute in COMPUTE_DTYPE.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Buffers are not parameters - they won't be updated by optimizer
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.ones((out_features, 1), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @torch.no_grad()
    def quantize(self, weight_fp32):
        """Quantize floating point weights to INT8."""
        # Compute scale per row (per output neuron)
        # Use a small epsilon to avoid division by zero
        max_val = weight_fp32.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        self.scale.copy_(max_val / 127.0)
        # Quantize and round
        q_weight = (weight_fp32 / self.scale).round().clamp(-128, 127).to(torch.int8)
        self.weight_int8.copy_(q_weight)

    def forward(self, x):
        # x: (..., in_features)
        # Dequantize weight to the activation dtype
        # We do this on the fly so we don't need a special INT8 kernel
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, weight, self.bias.to(x.dtype) if self.bias is not None else None)

    @classmethod
    def from_float(cls, mod):
        """Convert a standard nn.Linear to Int8WeightOnlyLinear."""
        device = mod.weight.device
        new_mod = cls(mod.in_features, mod.out_features, mod.bias is not None).to(device)
        new_mod.quantize(mod.weight.data)
        if mod.bias is not None:
            new_mod.bias.data.copy_(mod.bias.data)
        return new_mod

def convert_to_int8(module):
    """
    Recursively replace all nn.Linear layers with Int8WeightOnlyLinear.
    Note: this is intended for inference-only or post-training quantization.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, Int8WeightOnlyLinear):
            setattr(module, name, Int8WeightOnlyLinear.from_float(child))
        else:
            convert_to_int8(child)
    return module
