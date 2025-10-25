"""
Runtime interpretation wrapper for nanochat models with SAEs.

Provides real-time feature tracking and steering during inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import json

from sae.config import SAEConfig
from sae.models import BaseSAE, create_sae


class InterpretableModel(nn.Module):
    """Wrapper around nanochat model that adds SAE-based interpretability.

    Allows real-time feature tracking and steering during inference.

    Example:
        >>> model = load_nanochat_model("models/d20/base_final.pt")
        >>> saes = load_saes("models/d20/saes/")
        >>> interp_model = InterpretableModel(model, saes)
        >>>
        >>> # Track features during inference
        >>> with interp_model.interpretation_enabled():
        ...     output = interp_model(input_ids)
        ...     features = interp_model.get_active_features()
        >>>
        >>> # Steer model by modifying feature activations
        >>> steered_output = interp_model.steer(
        ...     input_ids,
        ...     feature_id=("blocks.10.hook_resid_post", 4232),
        ...     strength=2.0
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        saes: Dict[str, BaseSAE],
        device: Optional[str] = None,
    ):
        """Initialize interpretable model.

        Args:
            model: Base nanochat model
            saes: Dictionary mapping hook points to trained SAEs
            device: Device to run on (defaults to model device)
        """
        super().__init__()
        self.model = model
        self.saes = nn.ModuleDict(saes)

        if device is None:
            device = str(model.get_device())
        self.device = device

        # Move SAEs to device
        for sae in self.saes.values():
            sae.to(device)

        # State for feature tracking
        self._interpretation_active = False
        self._active_features: Dict[str, torch.Tensor] = {}
        self._hook_handles = []

        # State for feature steering
        self._steering_active = False
        self._steering_config: Dict[str, Tuple[int, float]] = {}  # hook_point -> (feature_idx, strength)

    def forward(self, *args, **kwargs):
        """Forward pass through base model."""
        return self.model(*args, **kwargs)

    @contextmanager
    def interpretation_enabled(self):
        """Context manager to enable feature tracking.

        Usage:
            >>> with model.interpretation_enabled():
            ...     output = model(input_ids)
            ...     features = model.get_active_features()
        """
        self._enable_interpretation()
        try:
            yield self
        finally:
            self._disable_interpretation()

    def _enable_interpretation(self):
        """Enable feature tracking by attaching hooks."""
        if self._interpretation_active:
            return

        for hook_point, sae in self.saes.items():
            # Get module to hook
            module = self._get_module_from_hook_point(hook_point)

            # Create hook function
            def make_hook(hp, sae_model):
                def hook_fn(module, input, output):
                    # Get activation
                    if isinstance(output, tuple):
                        activation = output[0]
                    else:
                        activation = output

                    # Apply SAE to get features
                    # Handle different shapes
                    original_shape = activation.shape
                    if activation.ndim == 3:
                        B, T, D = activation.shape
                        activation_flat = activation.reshape(B * T, D)
                    else:
                        activation_flat = activation

                    with torch.no_grad():
                        features = sae_model.get_feature_activations(activation_flat)

                    # Store features
                    self._active_features[hp] = features

                return hook_fn

            handle = module.register_forward_hook(make_hook(hook_point, sae))
            self._hook_handles.append(handle)

        self._interpretation_active = True

    def _disable_interpretation(self):
        """Disable feature tracking by removing hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._active_features = {}
        self._interpretation_active = False

    @contextmanager
    def steering_enabled(self, steering_config: Dict[str, Tuple[int, float]]):
        """Context manager to enable feature steering.

        Args:
            steering_config: Dict mapping hook points to (feature_idx, strength) tuples

        Usage:
            >>> steering = {
            ...     "blocks.10.hook_resid_post": (4232, 2.0),  # Amplify feature 4232
            ... }
            >>> with model.steering_enabled(steering):
            ...     output = model(input_ids)
        """
        self._enable_steering(steering_config)
        try:
            yield self
        finally:
            self._disable_steering()

    def _enable_steering(self, steering_config: Dict[str, Tuple[int, float]]):
        """Enable feature steering by attaching intervention hooks."""
        if self._steering_active:
            return

        self._steering_config = steering_config

        for hook_point, (feature_idx, strength) in steering_config.items():
            if hook_point not in self.saes:
                raise ValueError(f"No SAE for hook point: {hook_point}")

            module = self._get_module_from_hook_point(hook_point)
            sae = self.saes[hook_point]

            def make_steering_hook(sae_model, feat_idx, steer_strength):
                def hook_fn(module, input, output):
                    # Get activation
                    if isinstance(output, tuple):
                        activation = output[0]
                        rest = output[1:]
                    else:
                        activation = output
                        rest = ()

                    # Reshape if needed
                    original_shape = activation.shape
                    if activation.ndim == 3:
                        B, T, D = activation.shape
                        activation = activation.reshape(B * T, D)
                    else:
                        B, T, D = None, None, None

                    # Get current features
                    with torch.no_grad():
                        features = sae_model.get_feature_activations(activation)

                        # Modify feature
                        features[:, feat_idx] *= steer_strength

                        # Reconstruct with modified features
                        steered_activation = sae_model.decode(features)

                    # Reshape back
                    if B is not None and T is not None:
                        steered_activation = steered_activation.reshape(B, T, D)

                    # Return modified output
                    if rest:
                        return (steered_activation,) + rest
                    else:
                        return steered_activation

                return hook_fn

            handle = module.register_forward_hook(
                make_steering_hook(sae, feature_idx, strength)
            )
            self._hook_handles.append(handle)

        self._steering_active = True

    def _disable_steering(self):
        """Disable feature steering by removing hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._steering_config = {}
        self._steering_active = False

    def get_active_features(
        self,
        hook_point: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get active features from last forward pass.

        Args:
            hook_point: If specified, return features for this hook point only
            top_k: If specified, return only top-k most active features per example

        Returns:
            Dictionary mapping hook points to feature tensors
        """
        if not self._interpretation_active and not self._active_features:
            raise RuntimeError("No features available. Use interpretation_enabled() context manager.")

        if hook_point is not None:
            features = self._active_features.get(hook_point)
            if features is None:
                raise ValueError(f"No features for hook point: {hook_point}")
            result = {hook_point: features}
        else:
            result = self._active_features.copy()

        # Apply top-k filtering if requested
        if top_k is not None:
            for hp in result:
                features = result[hp]
                topk_values, topk_indices = torch.topk(features, k=min(top_k, features.shape[1]), dim=1)
                result[hp] = (topk_indices, topk_values)

        return result

    def steer(
        self,
        input_ids: torch.Tensor,
        feature_id: Tuple[str, int],
        strength: float,
        **kwargs
    ) -> torch.Tensor:
        """Run inference with feature steering.

        Args:
            input_ids: Input token IDs
            feature_id: Tuple of (hook_point, feature_idx)
            strength: Steering strength (multiplier for feature activation)
            **kwargs: Additional arguments to pass to model

        Returns:
            Model output with steered features
        """
        hook_point, feature_idx = feature_id
        steering_config = {hook_point: (feature_idx, strength)}

        with self.steering_enabled(steering_config):
            output = self.model(input_ids, **kwargs)

        return output

    def _get_module_from_hook_point(self, hook_point: str) -> nn.Module:
        """Get module from hook point string.

        Args:
            hook_point: Hook point (e.g., "blocks.10.hook_resid_post")

        Returns:
            Module to attach hook to
        """
        parts = hook_point.split(".")
        if parts[0] != "blocks":
            raise ValueError(f"Invalid hook point: {hook_point}")

        layer_idx = int(parts[1])
        hook_type = ".".join(parts[2:])

        block = self.model.transformer.h[layer_idx]

        if "hook_resid" in hook_type:
            return block
        elif "attn" in hook_type:
            return block.attn
        elif "mlp" in hook_type:
            return block.mlp
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")


def load_saes(
    sae_dir: Path,
    device: str = "cpu",
    hook_points: Optional[List[str]] = None,
) -> Dict[str, BaseSAE]:
    """Load trained SAEs from directory.

    Args:
        sae_dir: Directory containing SAE checkpoints
        device: Device to load SAEs on
        hook_points: If specified, only load SAEs for these hook points

    Returns:
        Dictionary mapping hook points to SAE models
    """
    sae_dir = Path(sae_dir)
    if not sae_dir.exists():
        raise ValueError(f"SAE directory does not exist: {sae_dir}")

    saes = {}

    # Find all SAE checkpoints
    checkpoint_files = list(sae_dir.glob("**/best_model.pt")) + list(sae_dir.glob("**/checkpoint_*.pt"))

    # Also look for direct .pt files
    if not checkpoint_files:
        checkpoint_files = list(sae_dir.glob("*.pt"))

    # Load each SAE
    for checkpoint_path in checkpoint_files:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get config
        if "config" in checkpoint:
            config = SAEConfig.from_dict(checkpoint["config"])
        else:
            # Try to load config from JSON
            config_path = checkpoint_path.parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = SAEConfig.from_dict(json.load(f))
            else:
                print(f"Warning: no config found for {checkpoint_path}, skipping")
                continue

        hook_point = config.hook_point

        # Filter by hook points if specified
        if hook_points is not None and hook_point not in hook_points:
            continue

        # Create SAE and load weights
        sae = create_sae(config)
        sae.load_state_dict(checkpoint["sae_state_dict"])
        sae.to(device)
        sae.eval()

        saes[hook_point] = sae
        print(f"Loaded SAE for {hook_point} from {checkpoint_path}")

    if not saes:
        print(f"Warning: no SAEs found in {sae_dir}")

    return saes


def save_sae(
    sae: BaseSAE,
    config: SAEConfig,
    save_path: Path,
    **metadata
):
    """Save SAE model and config.

    Args:
        sae: SAE model to save
        config: SAE configuration
        save_path: Path to save checkpoint
        **metadata: Additional metadata to include
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "sae_state_dict": sae.state_dict(),
        "config": config.to_dict(),
        **metadata
    }

    torch.save(checkpoint, save_path)

    # Also save config as JSON
    config_path = save_path.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"Saved SAE to {save_path}")
