"""
Mamba block implementation using Selective State Space Models.

This block provides an alternative to transformer attention that scales linearly
with sequence length rather than quadratically.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from nanochat.blocks import BaseBlock


def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    import torch.nn.functional as F
    return F.rms_norm(x, (x.size(-1),))


class MambaBlock(BaseBlock):
    """
    Mamba block using Selective State Space Model (S6).
    
    Architecture:
        x -> norm -> Mamba(SSM) -> residual
        [optional] x -> norm -> MLP -> residual
    
    Features:
    - Linear complexity in sequence length O(n) vs O(n²) for attention
    - Selective scan mechanism with input-dependent parameters
    - Hardware-aware implementation with fused CUDA kernels
    - Much smaller inference cache than KV-cache
    - No explicit position encodings needed (implicit in state evolution)
    
    Key differences from Transformer:
    - No rotary embeddings needed
    - No KV cache (uses state cache instead, much smaller)
    - Better for long sequences
    - Slightly different information flow
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError(
                "mamba-ssm package is required for MambaBlock. "
                "Install it with: pip install mamba-ssm>=2.0.0"
            )
        
        # Initialize Mamba SSM layer
        self.mixer = Mamba(
            d_model=config.n_embd,
            d_state=getattr(config, 'mamba_d_state', 16),
            d_conv=getattr(config, 'mamba_d_conv', 4),
            expand=getattr(config, 'mamba_expand', 2),
            dt_rank="auto",  # Auto-calculate based on d_model
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Use optimized CUDA kernels
            layer_idx=layer_idx,
            device=None,  # Will be moved to device by model
            dtype=torch.bfloat16,
        )
        
        # Optional MLP (Mamba already has gating, so this might be redundant)
        mamba_use_mlp = getattr(config, 'mamba_use_mlp', False)
        if mamba_use_mlp:
            from nanochat.gpt import MLP
            self.mlp = MLP(config)
        else:
            self.mlp = None
    
    def forward(self, x, context: Optional[Dict[str, Any]] = None):
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            context: Optional dictionary containing:
                - "inference_params": For stateful generation (Mamba-specific)
                Note: Mamba does NOT use cos_sin or kv_cache
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        if context is None:
            context = {}
        
        inference_params = context.get("inference_params", None)
        
        # Selective SSM with residual and pre-norm
        x = x + self.mixer(norm(x), inference_params=inference_params)
        
        # Optional MLP with residual
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        
        return x

