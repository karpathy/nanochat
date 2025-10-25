"""
Configuration for Sparse Autoencoders (SAEs).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SAEConfig:
    """Configuration for training and using Sparse Autoencoders.

    Attributes:
        d_in: Input dimension (typically model d_model)
        d_sae: SAE hidden dimension (expansion factor * d_in)
        activation: SAE activation function ("topk" or "relu")
        k: Number of active features for TopK activation
        l1_coefficient: L1 sparsity penalty for ReLU activation
        normalize_decoder: Whether to normalize decoder weights (recommended)
        dtype: Data type for SAE weights
        hook_point: Layer/component to hook (e.g., "blocks.10.hook_resid_post")
        expansion_factor: Expansion factor for hidden dimension (used if d_sae not specified)
    """

    # Model architecture
    d_in: int
    d_sae: Optional[int] = None
    activation: Literal["topk", "relu", "gated"] = "topk"

    # Sparsity control
    k: int = 64  # Number of active features for TopK
    l1_coefficient: float = 1e-3  # L1 penalty for ReLU

    # Training hyperparameters
    normalize_decoder: bool = True
    dtype: str = "bfloat16"

    # Hook configuration
    hook_point: str = "blocks.0.hook_resid_post"
    expansion_factor: int = 8

    # Training data
    num_activations: int = 10_000_000  # Number of activations to collect
    batch_size: int = 4096
    num_epochs: int = 10
    learning_rate: float = 3e-4
    warmup_steps: int = 1000

    # Dead latent resampling
    dead_latent_threshold: float = 0.001  # Fraction of activations where feature must activate
    resample_interval: int = 25000  # Steps between resampling checks

    # Evaluation
    eval_every: int = 1000  # Steps between evaluations
    save_every: int = 10000  # Steps between checkpoints

    def __post_init__(self):
        """Compute derived values."""
        if self.d_sae is None:
            self.d_sae = self.d_in * self.expansion_factor

    @classmethod
    def from_model(cls, model, layer_idx: int, hook_type: str = "resid_post", **kwargs):
        """Create SAE config from nanochat model.

        Args:
            model: Nanochat GPT model
            layer_idx: Layer index to hook (0 to n_layer-1)
            hook_type: Type of hook ("resid_post", "attn", "mlp")
            **kwargs: Additional configuration overrides

        Returns:
            SAEConfig instance
        """
        d_in = model.config.n_embd
        hook_point = f"blocks.{layer_idx}.hook_{hook_type}"

        return cls(
            d_in=d_in,
            hook_point=hook_point,
            **kwargs
        )

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation": self.activation,
            "k": self.k,
            "l1_coefficient": self.l1_coefficient,
            "normalize_decoder": self.normalize_decoder,
            "dtype": self.dtype,
            "hook_point": self.hook_point,
            "expansion_factor": self.expansion_factor,
        }

    @classmethod
    def from_dict(cls, d):
        """Load config from dictionary."""
        # Only keep keys that are in the config
        valid_keys = {
            "d_in", "d_sae", "activation", "k", "l1_coefficient",
            "normalize_decoder", "dtype", "hook_point", "expansion_factor",
            "num_activations", "batch_size", "num_epochs", "learning_rate",
            "warmup_steps", "dead_latent_threshold", "resample_interval",
            "eval_every", "save_every"
        }
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)
