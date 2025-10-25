"""
SAE training and evaluation.

This module provides a trainer for Sparse Autoencoders with support for:
- Dead latent resampling
- Learning rate warmup and decay
- Evaluation metrics
- Checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import json

from sae.config import SAEConfig
from sae.models import BaseSAE, create_sae
from sae.evaluator import SAEEvaluator


class SAETrainer:
    """Trainer for Sparse Autoencoders.

    Handles training loop, optimization, dead latent resampling, and checkpointing.
    """

    def __init__(
        self,
        sae: BaseSAE,
        config: SAEConfig,
        activations: torch.Tensor,
        val_activations: Optional[torch.Tensor] = None,
        device: str = "cuda",
        save_dir: Optional[Path] = None,
    ):
        """Initialize SAE trainer.

        Args:
            sae: SAE model to train
            config: SAE configuration
            activations: Training activations, shape (num_activations, d_in)
            val_activations: Optional validation activations
            device: Device to train on
            save_dir: Directory to save checkpoints and logs
        """
        self.sae = sae.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None

        # Create dataloaders
        self.train_loader = self._create_dataloader(activations, shuffle=True)
        self.val_loader = None
        if val_activations is not None:
            self.val_loader = self._create_dataloader(val_activations, shuffle=False)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Evaluator
        self.evaluator = SAEEvaluator(sae, config)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Statistics for dead latent detection
        self.feature_counts = torch.zeros(config.d_sae, device=device)

        # Logging
        self.train_losses = []
        self.val_losses = []

    def _create_dataloader(self, activations: torch.Tensor, shuffle: bool) -> DataLoader:
        """Create dataloader from activations."""
        dataset = TensorDataset(activations)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Keep simple for now
            pin_memory=True if self.device == "cuda" else False,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            verbose: Whether to show progress bar

        Returns:
            Dictionary of average training metrics
        """
        self.sae.train()
        epoch_metrics = {
            "mse_loss": 0.0,
            "l0": 0.0,
            "total_loss": 0.0,
        }
        num_batches = 0

        iterator = tqdm(self.train_loader, desc=f"Epoch {self.epoch}") if verbose else self.train_loader

        for batch in iterator:
            x = batch[0].to(self.device)

            # Forward pass
            reconstruction, features, metrics = self.sae(x)

            # Backward pass
            loss = metrics["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Normalize decoder weights if configured
            if self.config.normalize_decoder:
                self.sae.normalize_decoder_weights()

            # Update feature counts for dead latent detection
            with torch.no_grad():
                active_features = (features != 0).float().sum(dim=0)
                self.feature_counts += active_features

            # Log metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key].item()
            num_batches += 1

            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    "loss": f"{metrics['total_loss'].item():.4f}",
                    "l0": f"{metrics['l0'].item():.1f}",
                })

            # Periodic evaluation and checkpointing
            self.step += 1

            if self.step % self.config.eval_every == 0:
                self._evaluate_and_log()

            if self.step % self.config.save_every == 0:
                self._save_checkpoint()

            # Dead latent resampling
            if self.step % self.config.resample_interval == 0:
                self._resample_dead_latents()

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        self.epoch += 1
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate SAE on validation set.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.sae.eval()
        val_metrics = {
            "mse_loss": 0.0,
            "l0": 0.0,
            "total_loss": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            x = batch[0].to(self.device)
            reconstruction, features, metrics = self.sae(x)

            for key in val_metrics:
                if key in metrics:
                    val_metrics[key] += metrics[key].item()
            num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def _evaluate_and_log(self):
        """Evaluate and log metrics."""
        val_metrics = self.evaluate()
        if val_metrics:
            print(f"\nStep {self.step} - Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Track best model
            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self._save_checkpoint(is_best=True)

            self.val_losses.append(val_metrics)

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        if self.save_dir is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "sae_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "feature_counts": self.feature_counts.cpu(),
            "best_val_loss": self.best_val_loss,
        }

        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_step{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save as best if applicable
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save config as JSON for easy inspection
        config_path = self.save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _resample_dead_latents(self):
        """Resample dead latents that rarely activate.

        Dead latents are features that activate on fewer than a threshold fraction
        of training examples. We resample them by reinitializing their weights.
        """
        # Calculate activation frequency
        total_samples = self.step * self.config.batch_size
        activation_freq = self.feature_counts / total_samples

        # Find dead latents
        dead_mask = activation_freq < self.config.dead_latent_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            print(f"\nResampling {num_dead} dead latents (threshold: {self.config.dead_latent_threshold})")

            # Reinitialize dead latent weights
            with torch.no_grad():
                # Sample from active latents with high loss
                # For simplicity, just reinitialize randomly
                dead_indices = torch.where(dead_mask)[0]

                for idx in dead_indices:
                    # Reinitialize encoder weights
                    nn.init.kaiming_uniform_(self.sae.W_enc[:, idx].unsqueeze(1))
                    self.sae.b_enc[idx] = 0.0

                    # Reinitialize decoder weights
                    nn.init.kaiming_uniform_(self.sae.W_dec[idx].unsqueeze(0))

                # Normalize decoder if configured
                if self.config.normalize_decoder:
                    self.sae.normalize_decoder_weights()

            # Reset feature counts for resampled latents
            self.feature_counts[dead_mask] = 0

    def train(self, num_epochs: Optional[int] = None, verbose: bool = True):
        """Train SAE for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train. If None, uses config.num_epochs
            verbose: Whether to show progress bars
        """
        num_epochs = num_epochs or self.config.num_epochs

        for _ in range(num_epochs):
            epoch_metrics = self.train_epoch(verbose=verbose)

            if verbose:
                print(f"Epoch {self.epoch - 1} summary:")
                for key, value in epoch_metrics.items():
                    print(f"  {key}: {value:.4f}")

            self.train_losses.append(epoch_metrics)

        # Final evaluation and checkpoint
        self._evaluate_and_log()
        self._save_checkpoint()

        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.sae.load_state_dict(checkpoint["sae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.feature_counts = checkpoint["feature_counts"].to(self.device)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        print(f"Loaded checkpoint from step {self.step}, epoch {self.epoch}")


def train_sae_from_activations(
    activations: torch.Tensor,
    config: SAEConfig,
    val_split: float = 0.1,
    device: str = "cuda",
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[BaseSAE, SAETrainer]:
    """Train an SAE from activations.

    Convenience function that creates an SAE, splits data, and trains.

    Args:
        activations: Training activations, shape (num_activations, d_in)
        config: SAE configuration
        val_split: Fraction of data to use for validation
        device: Device to train on
        save_dir: Directory to save checkpoints
        verbose: Whether to show progress

    Returns:
        Tuple of (trained_sae, trainer)
    """
    # Split data into train/val
    num_val = int(len(activations) * val_split)
    indices = torch.randperm(len(activations))

    train_activations = activations[indices[num_val:]]
    val_activations = activations[indices[:num_val]] if num_val > 0 else None

    print(f"Training on {len(train_activations)} activations, validating on {num_val}")

    # Create SAE
    sae = create_sae(config)

    # Create trainer
    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=train_activations,
        val_activations=val_activations,
        device=device,
        save_dir=save_dir,
    )

    # Train
    trainer.train(verbose=verbose)

    return sae, trainer
