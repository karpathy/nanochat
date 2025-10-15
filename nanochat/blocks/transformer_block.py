"""
Transformer block implementation.

This is the original nanochat transformer architecture, refactored to fit the BaseBlock interface.
"""

from typing import Optional, Dict, Any
import torch.nn as nn

from nanochat.blocks import BaseBlock


# Import components from gpt.py
# We'll need to ensure these are accessible
def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    import torch.nn.functional as F
    return F.rms_norm(x, (x.size(-1),))


class TransformerBlock(BaseBlock):
    """
    Transformer block with Multi-Query Attention and MLP.
    
    Architecture:
        x -> norm -> CausalSelfAttention -> residual
        x -> norm -> MLP -> residual
    
    Features:
    - Rotary position embeddings (RoPE)
    - QK normalization
    - Multi-Query Attention (MQA)
    - ReLU² activation in MLP
    - Pre-normalization
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Import here to avoid circular dependency
        from nanochat.gpt import CausalSelfAttention, MLP
        
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x, context: Optional[Dict[str, Any]] = None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            context: Dictionary containing:
                - "cos_sin": Tuple of (cos, sin) for rotary embeddings
                - "kv_cache": Optional KV cache for inference
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        if context is None:
            context = {}
        
        cos_sin = context.get("cos_sin", None)
        kv_cache = context.get("kv_cache", None)
        
        # Self-attention with residual
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        
        # MLP with residual
        x = x + self.mlp(norm(x))
        
        return x

