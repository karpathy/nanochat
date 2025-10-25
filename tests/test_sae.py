"""
Basic tests for SAE implementation.

Run with: python -m pytest tests/test_sae.py -v
Or simply: python tests/test_sae.py
"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.config import SAEConfig
from sae.models import TopKSAE, ReLUSAE, GatedSAE, create_sae
from sae.hooks import ActivationCollector
from sae.trainer import SAETrainer
from sae.evaluator import SAEEvaluator
from sae.runtime import InterpretableModel


def test_sae_config():
    """Test SAE configuration."""
    config = SAEConfig(
        d_in=128,
        d_sae=1024,
        activation="topk",
        k=16,
    )

    assert config.d_in == 128
    assert config.d_sae == 1024
    assert config.expansion_factor == 8

    # Test dict conversion
    config_dict = config.to_dict()
    config2 = SAEConfig.from_dict(config_dict)
    assert config2.d_in == config.d_in
    assert config2.d_sae == config.d_sae

    print("✓ SAEConfig tests passed")


def test_topk_sae():
    """Test TopK SAE forward pass."""
    config = SAEConfig(
        d_in=128,
        d_sae=1024,
        activation="topk",
        k=16,
    )

    sae = TopKSAE(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.d_in)

    reconstruction, features, metrics = sae(x)

    assert reconstruction.shape == x.shape
    assert features.shape == (batch_size, config.d_sae)
    assert "mse_loss" in metrics
    assert "l0" in metrics

    # Check sparsity
    l0 = (features != 0).sum(dim=-1).float().mean().item()
    assert abs(l0 - config.k) < 1.0, f"Expected L0≈{config.k}, got {l0}"

    print("✓ TopK SAE tests passed")


def test_relu_sae():
    """Test ReLU SAE forward pass."""
    config = SAEConfig(
        d_in=128,
        d_sae=1024,
        activation="relu",
        l1_coefficient=1e-3,
    )

    sae = ReLUSAE(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.d_in)

    reconstruction, features, metrics = sae(x)

    assert reconstruction.shape == x.shape
    assert features.shape == (batch_size, config.d_sae)
    assert "mse_loss" in metrics
    assert "l1_loss" in metrics
    assert "total_loss" in metrics

    # Check features are non-negative (ReLU)
    assert (features >= 0).all()

    print("✓ ReLU SAE tests passed")


def test_gated_sae():
    """Test Gated SAE forward pass."""
    config = SAEConfig(
        d_in=128,
        d_sae=1024,
        activation="gated",
        l1_coefficient=1e-3,
    )

    sae = GatedSAE(config)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.d_in)

    reconstruction, features, metrics = sae(x)

    assert reconstruction.shape == x.shape
    assert features.shape == (batch_size, config.d_sae)
    assert "mse_loss" in metrics
    assert "l0" in metrics

    print("✓ Gated SAE tests passed")


def test_sae_factory():
    """Test SAE factory function."""
    # TopK
    config_topk = SAEConfig(d_in=128, activation="topk")
    sae_topk = create_sae(config_topk)
    assert isinstance(sae_topk, TopKSAE)

    # ReLU
    config_relu = SAEConfig(d_in=128, activation="relu")
    sae_relu = create_sae(config_relu)
    assert isinstance(sae_relu, ReLUSAE)

    # Gated
    config_gated = SAEConfig(d_in=128, activation="gated")
    sae_gated = create_sae(config_gated)
    assert isinstance(sae_gated, GatedSAE)

    print("✓ SAE factory tests passed")


def test_sae_training():
    """Test SAE training loop."""
    # Create small SAE
    config = SAEConfig(
        d_in=64,
        d_sae=256,
        activation="topk",
        k=16,
        batch_size=32,
        num_epochs=2,
    )

    sae = TopKSAE(config)

    # Generate random training data
    num_samples = 1000
    activations = torch.randn(num_samples, config.d_in)
    val_activations = torch.randn(200, config.d_in)

    # Create trainer
    trainer = SAETrainer(
        sae=sae,
        config=config,
        activations=activations,
        val_activations=val_activations,
        device="cpu",
    )

    # Train for 2 epochs
    initial_loss = None
    for epoch in range(2):
        metrics = trainer.train_epoch(verbose=False)
        if initial_loss is None:
            initial_loss = metrics["total_loss"]

    # Loss should decrease
    final_loss = metrics["total_loss"]
    assert final_loss < initial_loss, "Loss should decrease during training"

    print("✓ SAE training tests passed")


def test_sae_evaluator():
    """Test SAE evaluator."""
    config = SAEConfig(
        d_in=64,
        d_sae=256,
        activation="topk",
        k=16,
    )

    sae = TopKSAE(config)

    # Generate test data
    test_activations = torch.randn(500, config.d_in)

    # Create evaluator
    evaluator = SAEEvaluator(sae, config)

    # Evaluate
    metrics = evaluator.evaluate(test_activations, compute_dead_latents=True)

    assert metrics.mse_loss >= 0
    assert 0 <= metrics.explained_variance <= 1
    assert metrics.l0_mean > 0
    assert 0 <= metrics.dead_latent_fraction <= 1

    print("✓ SAE evaluator tests passed")


def test_activation_collector():
    """Test activation collection with hooks."""
    # Create a simple model (Linear layer)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
    )

    # Collect activations from the ReLU layer
    collector = ActivationCollector(
        model=model,
        hook_points=["1"],  # Index of ReLU layer
        max_activations=100,
        device="cpu",
    )

    with collector:
        for _ in range(10):
            x = torch.randn(10, 64)
            _ = model(x)

    activations = collector.get_activations()
    assert "1" in activations
    assert activations["1"].shape[0] == 100
    assert activations["1"].shape[1] == 128

    print("✓ Activation collector tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("Running SAE Implementation Tests")
    print("="*80 + "\n")

    try:
        test_sae_config()
        test_topk_sae()
        test_relu_sae()
        test_gated_sae()
        test_sae_factory()
        test_sae_training()
        test_sae_evaluator()
        test_activation_collector()

        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
