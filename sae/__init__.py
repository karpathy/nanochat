"""
SAE-based interpretability extension for nanochat.

This module provides Sparse Autoencoder (SAE) functionality for mechanistic interpretability
of nanochat models. It includes:
- SAE model architectures (TopK, ReLU, Gated)
- Activation collection via PyTorch hooks
- SAE training and evaluation
- Runtime interpretation and feature steering
- Neuronpedia integration
"""

from sae.config import SAEConfig
from sae.models import TopKSAE, ReLUSAE
from sae.hooks import ActivationCollector
from sae.runtime import InterpretableModel, load_saes

__all__ = [
    "SAEConfig",
    "TopKSAE",
    "ReLUSAE",
    "ActivationCollector",
    "InterpretableModel",
    "load_saes",
]
