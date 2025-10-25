"""
Sparse Autoencoder (SAE) model architectures.

Implements TopK, ReLU, and Gated SAE variants for mechanistic interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from sae.config import SAEConfig


class BaseSAE(nn.Module):
    """Base class for Sparse Autoencoders."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.d_in = config.d_in
        self.d_sae = config.d_sae

        # Encoder: d_in -> d_sae
        self.W_enc = nn.Parameter(torch.empty(config.d_in, config.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(config.d_sae))

        # Decoder: d_sae -> d_in
        self.W_dec = nn.Parameter(torch.empty(config.d_sae, config.d_in))
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize SAE weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.W_enc, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_uniform_(self.W_dec, a=0, mode='fan_in', nonlinearity='linear')

        # Normalize decoder weights if specified
        if self.config.normalize_decoder:
            self.normalize_decoder_weights()

    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm (critical for stability)."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1, p=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation."""
        return x @ self.W_enc + self.b_enc

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation to reconstruction."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through SAE.

        Args:
            x: Input activations, shape (batch, d_in)

        Returns:
            Tuple of (reconstruction, hidden_features, metrics_dict)
        """
        raise NotImplementedError("Subclasses must implement forward()")


class TopKSAE(BaseSAE):
    """TopK Sparse Autoencoder.

    Uses TopK activation to select k most active features, providing direct
    sparsity control without L1 penalty tuning. Recommended by OpenAI's scaling work.

    Reference: https://arxiv.org/abs/2406.04093
    """

    def __init__(self, config: SAEConfig):
        super().__init__(config)
        assert config.activation == "topk", f"TopKSAE requires activation='topk', got {config.activation}"
        self.k = config.k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass with TopK activation.

        Args:
            x: Input activations, shape (batch, d_in)

        Returns:
            reconstruction: Reconstructed activations, shape (batch, d_in)
            features: Sparse feature activations, shape (batch, d_sae)
            metrics: Dict with loss components and statistics
        """
        # Encode
        pre_activation = self.encode(x)

        # TopK activation: keep top k features, zero out rest
        topk_values, topk_indices = torch.topk(pre_activation, k=self.k, dim=-1)
        features = torch.zeros_like(pre_activation)
        features.scatter_(-1, topk_indices, topk_values)

        # Decode
        reconstruction = self.decode(features)

        # Compute metrics
        mse_loss = F.mse_loss(reconstruction, x)
        l0 = (features != 0).float().sum(dim=-1).mean()  # Average number of active features

        metrics = {
            "mse_loss": mse_loss,
            "l0": l0,
            "total_loss": mse_loss,  # TopK doesn't need L1 penalty
        }

        return reconstruction, features, metrics

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input (inference mode)."""
        pre_activation = self.encode(x)
        topk_values, topk_indices = torch.topk(pre_activation, k=self.k, dim=-1)
        features = torch.zeros_like(pre_activation)
        features.scatter_(-1, topk_indices, topk_values)
        return features


class ReLUSAE(BaseSAE):
    """ReLU Sparse Autoencoder.

    Traditional SAE with ReLU activation and L1 sparsity penalty.
    Requires tuning the L1 coefficient to balance reconstruction vs sparsity.

    Reference: https://transformer-circuits.pub/2023/monosemantic-features
    """

    def __init__(self, config: SAEConfig):
        super().__init__(config)
        assert config.activation == "relu", f"ReLUSAE requires activation='relu', got {config.activation}"
        self.l1_coefficient = config.l1_coefficient

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass with ReLU activation and L1 penalty.

        Args:
            x: Input activations, shape (batch, d_in)

        Returns:
            reconstruction: Reconstructed activations, shape (batch, d_in)
            features: Sparse feature activations, shape (batch, d_sae)
            metrics: Dict with loss components and statistics
        """
        # Encode and activate
        pre_activation = self.encode(x)
        features = F.relu(pre_activation)

        # Decode
        reconstruction = self.decode(features)

        # Compute losses
        mse_loss = F.mse_loss(reconstruction, x)
        l1_loss = features.abs().sum(dim=-1).mean()
        total_loss = mse_loss + self.l1_coefficient * l1_loss

        # Compute statistics
        l0 = (features != 0).float().sum(dim=-1).mean()

        metrics = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "l0": l0,
            "total_loss": total_loss,
        }

        return reconstruction, features, metrics

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input (inference mode)."""
        pre_activation = self.encode(x)
        return F.relu(pre_activation)


class GatedSAE(BaseSAE):
    """Gated Sparse Autoencoder.

    Separates feature selection (gating) from feature magnitude.
    Can learn better sparse representations but is more complex.

    Reference: https://arxiv.org/abs/2404.16014
    """

    def __init__(self, config: SAEConfig):
        super().__init__(config)
        assert config.activation == "gated", f"GatedSAE requires activation='gated', got {config.activation}"

        # Additional gating parameters
        self.W_gate = nn.Parameter(torch.empty(config.d_in, config.d_sae))
        self.b_gate = nn.Parameter(torch.zeros(config.d_sae))

        # Initialize gating weights
        nn.init.kaiming_uniform_(self.W_gate, a=0, mode='fan_in', nonlinearity='linear')

        self.l1_coefficient = config.l1_coefficient

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass with gated activation.

        Args:
            x: Input activations, shape (batch, d_in)

        Returns:
            reconstruction: Reconstructed activations, shape (batch, d_in)
            features: Sparse feature activations, shape (batch, d_sae)
            metrics: Dict with loss components and statistics
        """
        # Magnitude pathway
        magnitude = self.encode(x)

        # Gating pathway
        gate_logits = x @ self.W_gate + self.b_gate
        gate = (gate_logits > 0).float()  # Binary gate

        # Combine: only keep features where gate is active
        features = magnitude * gate

        # Decode
        reconstruction = self.decode(features)

        # Compute losses
        mse_loss = F.mse_loss(reconstruction, x)
        # L1 penalty on gate activations encourages sparsity
        l1_loss = gate.sum(dim=-1).mean()
        total_loss = mse_loss + self.l1_coefficient * l1_loss

        # Statistics
        l0 = gate.sum(dim=-1).mean()

        metrics = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "l0": l0,
            "total_loss": total_loss,
        }

        return reconstruction, features, metrics

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input (inference mode)."""
        magnitude = self.encode(x)
        gate_logits = x @ self.W_gate + self.b_gate
        gate = (gate_logits > 0).float()
        return magnitude * gate


def create_sae(config: SAEConfig) -> BaseSAE:
    """Factory function to create SAE based on config.

    Args:
        config: SAE configuration

    Returns:
        SAE instance (TopKSAE, ReLUSAE, or GatedSAE)
    """
    if config.activation == "topk":
        return TopKSAE(config)
    elif config.activation == "relu":
        return ReLUSAE(config)
    elif config.activation == "gated":
        return GatedSAE(config)
    else:
        raise ValueError(f"Unknown activation type: {config.activation}")
