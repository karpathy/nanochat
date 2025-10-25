"""
Activation collection using PyTorch forward hooks.

This module provides utilities to collect intermediate activations from nanochat
models for SAE training, using minimal memory and performance overhead.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm


class ActivationCollector:
    """Collects activations from specified hook points in a model.

    Uses PyTorch forward hooks to capture intermediate activations during
    model execution. Activations are stored in memory and can be saved to disk.

    Example:
        >>> collector = ActivationCollector(
        ...     model,
        ...     hook_points=["blocks.10.hook_resid_post", "blocks.15.hook_resid_post"],
        ...     max_activations=1_000_000
        ... )
        >>> with collector:
        ...     for batch in dataloader:
        ...         model(batch)
        >>> activations = collector.get_activations()
    """

    def __init__(
        self,
        model: nn.Module,
        hook_points: List[str],
        max_activations: int = 10_000_000,
        device: str = "cpu",
        save_path: Optional[Path] = None,
    ):
        """Initialize activation collector.

        Args:
            model: PyTorch model to collect activations from
            hook_points: List of hook point names (e.g., "blocks.10.hook_resid_post")
            max_activations: Maximum number of activations to collect per hook point
            device: Device to store activations on ("cpu" or "cuda")
            save_path: Optional path to save activations to disk
        """
        self.model = model
        self.hook_points = hook_points
        self.max_activations = max_activations
        self.device = device
        self.save_path = Path(save_path) if save_path else None

        # Storage for activations
        self.activations: Dict[str, List[torch.Tensor]] = {hp: [] for hp in hook_points}
        self.counts: Dict[str, int] = {hp: 0 for hp in hook_points}

        # Hook handles (for cleanup)
        self.handles = []

    def _get_hook_fn(self, hook_point: str) -> Callable:
        """Create a hook function for a specific hook point.

        Args:
            hook_point: Name of the hook point

        Returns:
            Hook function that captures activations
        """
        def hook_fn(module, input, output):
            # Check if we've collected enough activations
            if self.counts[hook_point] >= self.max_activations:
                return

            # Get the activation tensor
            # Output can be a tuple (output, kv_cache) or just output
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Flatten batch and sequence dimensions: (B, T, D) -> (B*T, D)
            if activation.ndim == 3:
                B, T, D = activation.shape
                activation = activation.reshape(B * T, D)
            elif activation.ndim == 2:
                # Already flattened
                pass
            else:
                raise ValueError(f"Unexpected activation shape: {activation.shape}")

            # Move to target device and detach
            activation = activation.detach().to(self.device)

            # Store activation
            num_new = activation.shape[0]
            remaining = self.max_activations - self.counts[hook_point]
            if num_new > remaining:
                activation = activation[:remaining]
                num_new = remaining

            self.activations[hook_point].append(activation)
            self.counts[hook_point] += num_new

        return hook_fn

    def _attach_hooks(self):
        """Attach forward hooks to the model."""
        for hook_point in self.hook_points:
            # Parse hook point to get module
            module = self._get_module_from_hook_point(hook_point)

            # Register forward hook
            handle = module.register_forward_hook(self._get_hook_fn(hook_point))
            self.handles.append(handle)

    def _get_module_from_hook_point(self, hook_point: str) -> nn.Module:
        """Get module from hook point string.

        Args:
            hook_point: Hook point string (e.g., "blocks.10.hook_resid_post")

        Returns:
            Module to attach hook to
        """
        # For nanochat, we need to hook at the Block level
        # Hook points look like: "blocks.{i}.hook_{type}"
        # We'll hook the entire block and capture the residual stream

        parts = hook_point.split(".")
        if parts[0] != "blocks":
            raise ValueError(f"Invalid hook point: {hook_point}. Must start with 'blocks.'")

        layer_idx = int(parts[1])
        hook_type = ".".join(parts[2:])  # e.g., "hook_resid_post", "attn.hook_result"

        # Get the block
        block = self.model.transformer.h[layer_idx]

        # For now, we'll just hook the entire block's output (residual stream)
        # More sophisticated hooks can be added later
        if "hook_resid" in hook_type:
            return block
        elif "attn" in hook_type:
            return block.attn
        elif "mlp" in hook_type:
            return block.mlp
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        """Context manager entry: attach hooks."""
        self._attach_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: remove hooks."""
        self._remove_hooks()

    def get_activations(self, hook_point: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get collected activations.

        Args:
            hook_point: If specified, return activations for this hook point only.
                       Otherwise, return all activations.

        Returns:
            Dictionary mapping hook points to activation tensors
        """
        if hook_point is not None:
            # Return activations for single hook point
            if hook_point not in self.activations:
                raise ValueError(f"Unknown hook point: {hook_point}")
            acts = torch.cat(self.activations[hook_point], dim=0)
            return {hook_point: acts}
        else:
            # Return all activations
            return {
                hp: torch.cat(acts, dim=0) if acts else torch.empty(0)
                for hp, acts in self.activations.items()
            }

    def save_activations(self, save_path: Optional[Path] = None):
        """Save collected activations to disk.

        Args:
            save_path: Path to save activations. If None, uses self.save_path
        """
        save_path = save_path or self.save_path
        if save_path is None:
            raise ValueError("No save path specified")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        for hook_point, acts in self.get_activations().items():
            # Sanitize hook point name for filename
            filename = hook_point.replace(".", "_") + ".pt"
            filepath = save_path / filename

            # Save as PyTorch tensor
            torch.save(acts, filepath)
            print(f"Saved {acts.shape[0]} activations for {hook_point} to {filepath}")

    @staticmethod
    def load_activations(load_path: Path, hook_points: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Load activations from disk.

        Args:
            load_path: Directory containing saved activations
            hook_points: If specified, only load these hook points

        Returns:
            Dictionary mapping hook points to activation tensors
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"Load path does not exist: {load_path}")

        activations = {}

        if hook_points is None:
            # Load all .pt files in directory
            pt_files = list(load_path.glob("*.pt"))
        else:
            # Load specific hook points
            pt_files = [
                load_path / (hp.replace(".", "_") + ".pt")
                for hp in hook_points
            ]

        for filepath in pt_files:
            if not filepath.exists():
                print(f"Warning: file not found: {filepath}")
                continue

            # Reconstruct hook point name from filename
            hook_point = filepath.stem.replace("_", ".")

            # Load tensor
            acts = torch.load(filepath)
            activations[hook_point] = acts
            print(f"Loaded {acts.shape[0]} activations for {hook_point}")

        return activations

    def clear(self):
        """Clear all collected activations."""
        self.activations = {hp: [] for hp in self.hook_points}
        self.counts = {hp: 0 for hp in self.hook_points}


def collect_activations_from_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    hook_points: List[str],
    max_activations: int = 10_000_000,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Collect activations from a dataloader.

    Convenience function that wraps ActivationCollector and iterates through
    a dataloader to collect activations.

    Args:
        model: PyTorch model
        dataloader: DataLoader providing input batches
        hook_points: List of hook points to collect activations from
        max_activations: Maximum number of activations to collect
        device: Device to store activations on
        save_path: Optional path to save activations
        verbose: Whether to show progress bar

    Returns:
        Dictionary mapping hook points to activation tensors
    """
    collector = ActivationCollector(
        model,
        hook_points=hook_points,
        max_activations=max_activations,
        device=device,
        save_path=save_path,
    )

    model.eval()  # Set model to eval mode
    with torch.no_grad(), collector:
        iterator = tqdm(dataloader, desc="Collecting activations") if verbose else dataloader

        for batch in iterator:
            # Check if we've collected enough
            if all(collector.counts[hp] >= max_activations for hp in hook_points):
                break

            # Move batch to model device
            if isinstance(batch, dict):
                batch = {k: v.to(model.get_device()) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(model.get_device()) if isinstance(x, torch.Tensor) else x for x in batch]
                model(*batch)
            else:
                batch = batch.to(model.get_device())
                model(batch)

    # Save if requested
    if save_path is not None:
        collector.save_activations()

    return collector.get_activations()
