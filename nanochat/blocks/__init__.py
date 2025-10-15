"""
Block abstractions for hybrid architectures.

This module provides a clean abstraction for different block types (Transformer, Mamba, etc.)
that can be mixed and matched in a single model.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn


class BaseBlock(nn.Module, ABC):
    """
    Abstract base class for all block types in the model.
    
    All blocks must implement:
    - forward(x, context): Process input with optional context
    - get_num_params(): Return number of parameters
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
    
    @abstractmethod
    def forward(self, x, context: Optional[Dict[str, Any]] = None):
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            context: Optional dictionary containing:
                - "cos_sin": Tuple of (cos, sin) for rotary embeddings (Transformer only)
                - "kv_cache": KV cache for inference (Transformer only)
                - "inference_params": Inference parameters (Mamba only)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        pass
    
    def get_num_params(self) -> int:
        """Return the number of parameters in this block."""
        return sum(p.numel() for p in self.parameters())
    
    def get_block_type(self) -> str:
        """Return a string identifier for the block type."""
        return self.__class__.__name__


def create_block(block_type: str, config, layer_idx: int) -> BaseBlock:
    """
    Factory function to create blocks based on type string.
    
    Args:
        block_type: One of:
            - "T" or "transformer": TransformerBlock
            - "M" or "mamba": MambaBlock
        config: Model configuration object
        layer_idx: Index of this block in the model
    
    Returns:
        Instance of the appropriate block type
    
    Raises:
        ValueError: If block_type is not recognized
    """
    from nanochat.blocks.transformer_block import TransformerBlock
    from nanochat.blocks.mamba_block import MambaBlock
    
    block_type = block_type.lower()
    
    if block_type in ("t", "transformer"):
        return TransformerBlock(config, layer_idx)
    elif block_type in ("m", "mamba"):
        return MambaBlock(config, layer_idx)
    else:
        raise ValueError(
            f"Unknown block type: {block_type}. "
            f"Valid types are: 'T'/'transformer', 'M'/'mamba'"
        )


__all__ = [
    "BaseBlock",
    "create_block",
]

