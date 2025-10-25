"""
SAE evaluation metrics.

Provides comprehensive evaluation of SAE quality including:
- Reconstruction quality (MSE, explained variance)
- Sparsity metrics (L0, dead latents)
- Feature interpretability (via sampling and analysis)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from dataclasses import dataclass

from sae.config import SAEConfig
from sae.models import BaseSAE


@dataclass
class SAEMetrics:
    """Container for SAE evaluation metrics."""

    # Reconstruction quality
    mse_loss: float
    explained_variance: float
    reconstruction_score: float  # 1 - MSE/variance

    # Sparsity metrics
    l0_mean: float  # Average number of active features
    l0_std: float
    l1_mean: float  # Average L1 norm of features
    dead_latent_fraction: float  # Fraction of features that never activate

    # Distribution stats
    max_activation: float
    mean_activation: float  # Mean of non-zero activations

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "mse_loss": self.mse_loss,
            "explained_variance": self.explained_variance,
            "reconstruction_score": self.reconstruction_score,
            "l0_mean": self.l0_mean,
            "l0_std": self.l0_std,
            "l1_mean": self.l1_mean,
            "dead_latent_fraction": self.dead_latent_fraction,
            "max_activation": self.max_activation,
            "mean_activation": self.mean_activation,
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        lines = [
            "SAE Evaluation Metrics",
            "=" * 50,
            "Reconstruction Quality:",
            f"  MSE Loss: {self.mse_loss:.6f}",
            f"  Explained Variance: {self.explained_variance:.4f}",
            f"  Reconstruction Score: {self.reconstruction_score:.4f}",
            "",
            "Sparsity:",
            f"  L0 (mean ± std): {self.l0_mean:.1f} ± {self.l0_std:.1f}",
            f"  L1 (mean): {self.l1_mean:.4f}",
            f"  Dead Latents: {self.dead_latent_fraction*100:.2f}%",
            "",
            "Activation Statistics:",
            f"  Max Activation: {self.max_activation:.4f}",
            f"  Mean Activation (non-zero): {self.mean_activation:.4f}",
        ]
        return "\n".join(lines)


class SAEEvaluator:
    """Evaluator for Sparse Autoencoders."""

    def __init__(self, sae: BaseSAE, config: SAEConfig):
        """Initialize evaluator.

        Args:
            sae: SAE model to evaluate
            config: SAE configuration
        """
        self.sae = sae
        self.config = config

    @torch.no_grad()
    def evaluate(
        self,
        activations: torch.Tensor,
        batch_size: int = 4096,
        compute_dead_latents: bool = True,
    ) -> SAEMetrics:
        """Evaluate SAE on activations.

        Args:
            activations: Activations to evaluate on, shape (num_activations, d_in)
            batch_size: Batch size for evaluation
            compute_dead_latents: Whether to compute dead latent statistics (slower)

        Returns:
            SAEMetrics object
        """
        self.sae.eval()
        device = next(self.sae.parameters()).device

        # Move activations to device in batches
        num_batches = (len(activations) + batch_size - 1) // batch_size

        # Accumulators
        total_mse = 0.0
        total_variance = 0.0
        l0_values = []
        l1_values = []
        max_activations = []
        mean_activations = []

        if compute_dead_latents:
            feature_counts = torch.zeros(self.config.d_sae, device=device)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(activations))
            batch = activations[start_idx:end_idx].to(device)

            # Forward pass
            reconstruction, features, _ = self.sae(batch)

            # Reconstruction quality
            mse = F.mse_loss(reconstruction, batch, reduction='sum').item()
            variance = batch.var(dim=0).sum().item() * batch.shape[0]

            total_mse += mse
            total_variance += variance

            # Sparsity metrics
            l0 = (features != 0).float().sum(dim=-1)
            l1 = features.abs().sum(dim=-1)

            l0_values.append(l0.cpu())
            l1_values.append(l1.cpu())

            # Activation statistics
            max_activations.append(features.max().item())
            non_zero_features = features[features != 0]
            if len(non_zero_features) > 0:
                mean_activations.append(non_zero_features.mean().item())

            # Dead latent tracking
            if compute_dead_latents:
                active = (features != 0).float().sum(dim=0)
                feature_counts += active

        # Compute final metrics
        mse_loss = total_mse / len(activations)
        variance = total_variance / len(activations)
        explained_variance = max(0.0, 1.0 - mse_loss / variance) if variance > 0 else 0.0
        reconstruction_score = 1.0 - mse_loss / variance if variance > 0 else 0.0

        l0_values = torch.cat(l0_values)
        l1_values = torch.cat(l1_values)

        l0_mean = l0_values.mean().item()
        l0_std = l0_values.std().item()
        l1_mean = l1_values.mean().item()

        max_activation = max(max_activations) if max_activations else 0.0
        mean_activation = sum(mean_activations) / len(mean_activations) if mean_activations else 0.0

        if compute_dead_latents:
            dead_latents = (feature_counts == 0).sum().item()
            dead_latent_fraction = dead_latents / self.config.d_sae
        else:
            dead_latent_fraction = 0.0

        return SAEMetrics(
            mse_loss=mse_loss,
            explained_variance=explained_variance,
            reconstruction_score=reconstruction_score,
            l0_mean=l0_mean,
            l0_std=l0_std,
            l1_mean=l1_mean,
            dead_latent_fraction=dead_latent_fraction,
            max_activation=max_activation,
            mean_activation=mean_activation,
        )

    @torch.no_grad()
    def get_top_activating_examples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        k: int = 10,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Get top-k activating examples for a specific feature.

        Args:
            feature_idx: Index of feature to analyze
            activations: Activations to search through
            k: Number of top examples to return
            batch_size: Batch size for processing

        Returns:
            Indices of top-k activating examples
        """
        self.sae.eval()
        device = next(self.sae.parameters()).device

        num_batches = (len(activations) + batch_size - 1) // batch_size
        all_feature_acts = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(activations))
            batch = activations[start_idx:end_idx].to(device)

            # Get feature activations
            _, features, _ = self.sae(batch)
            feature_acts = features[:, feature_idx]

            all_feature_acts.append(feature_acts.cpu())

        # Concatenate and get top-k
        all_feature_acts = torch.cat(all_feature_acts)
        topk_values, topk_indices = torch.topk(all_feature_acts, k=min(k, len(all_feature_acts)))

        return topk_indices

    @torch.no_grad()
    def analyze_feature(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        batch_size: int = 4096,
    ) -> Dict[str, float]:
        """Analyze a specific feature.

        Args:
            feature_idx: Index of feature to analyze
            activations: Activations to analyze over
            batch_size: Batch size for processing

        Returns:
            Dictionary of feature statistics
        """
        self.sae.eval()
        device = next(self.sae.parameters()).device

        num_batches = (len(activations) + batch_size - 1) // batch_size
        feature_acts = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(activations))
            batch = activations[start_idx:end_idx].to(device)

            # Get feature activations
            _, features, _ = self.sae(batch)
            feature_acts.append(features[:, feature_idx].cpu())

        feature_acts = torch.cat(feature_acts)

        # Compute statistics
        activation_freq = (feature_acts > 0).float().mean().item()
        mean_activation = feature_acts.mean().item()
        max_activation = feature_acts.max().item()
        std_activation = feature_acts.std().item()

        non_zero = feature_acts[feature_acts > 0]
        mean_when_active = non_zero.mean().item() if len(non_zero) > 0 else 0.0

        return {
            "activation_frequency": activation_freq,
            "mean_activation": mean_activation,
            "mean_when_active": mean_when_active,
            "max_activation": max_activation,
            "std_activation": std_activation,
        }

    @torch.no_grad()
    def get_feature_dashboard_data(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        top_k: int = 10,
    ) -> Dict:
        """Get comprehensive data for feature dashboard.

        Args:
            feature_idx: Feature index to analyze
            activations: Activations to analyze
            top_k: Number of top examples to return

        Returns:
            Dictionary with feature analysis data
        """
        stats = self.analyze_feature(feature_idx, activations)
        top_indices = self.get_top_activating_examples(feature_idx, activations, k=top_k)

        return {
            "feature_idx": feature_idx,
            "statistics": stats,
            "top_activating_indices": top_indices.tolist(),
        }
