"""
MLP (feedforward) layers for GPT model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.attention import Linear


class MLP(nn.Module):
    """MLP with ReLU^2 activation."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc -> relu^2 -> proj."""
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
