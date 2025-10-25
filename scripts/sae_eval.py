"""
Evaluate trained Sparse Autoencoders.

This script evaluates SAE quality and generates feature visualizations.

Usage:
    # Evaluate specific SAE
    python -m scripts.sae_eval --sae_path sae_outputs/layer_10/best_model.pt

    # Evaluate all SAEs in directory
    python -m scripts.sae_eval --sae_dir sae_outputs

    # Generate feature dashboards
    python -m scripts.sae_eval --sae_path sae_outputs/layer_10/best_model.pt \
        --generate_dashboards --top_k 20
"""

import argparse
import torch
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae.config import SAEConfig
from sae.models import create_sae
from sae.evaluator import SAEEvaluator
from sae.feature_viz import FeatureVisualizer, generate_sae_summary


def load_sae_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load SAE from checkpoint."""
    print(f"Loading SAE from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    if "config" in checkpoint:
        config = SAEConfig.from_dict(checkpoint["config"])
    else:
        # Try loading from JSON
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = SAEConfig.from_dict(json.load(f))
        else:
            raise ValueError("No config found in checkpoint")

    # Create and load SAE
    sae = create_sae(config)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    sae.to(device)
    sae.eval()

    print(f"Loaded SAE: {config.hook_point}, d_in={config.d_in}, d_sae={config.d_sae}")

    return sae, config


def generate_test_activations(d_in: int, num_samples: int = 10000, device: str = "cuda"):
    """Generate random test activations.

    In production, you would use real model activations.
    """
    # Generate random activations with some structure
    activations = torch.randn(num_samples, d_in, device=device)
    return activations


def evaluate_sae(
    sae,
    config: SAEConfig,
    activations: torch.Tensor,
    output_dir: Path,
    generate_dashboards: bool = False,
    top_k_features: int = 10,
):
    """Evaluate SAE and generate reports."""

    print(f"\nEvaluating SAE: {config.hook_point}")
    print(f"Using {activations.shape[0]} test activations")

    # Create evaluator
    evaluator = SAEEvaluator(sae, config)

    # Evaluate
    print("\nComputing evaluation metrics...")
    metrics = evaluator.evaluate(activations, compute_dead_latents=True)

    # Print metrics
    print("\n" + "="*80)
    print(str(metrics))
    print("="*80)

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Generate SAE summary
    print("\nGenerating SAE summary...")
    summary = generate_sae_summary(
        sae=sae,
        config=config,
        activations=activations,
        save_path=output_dir / "sae_summary.json",
    )

    # Generate feature dashboards if requested
    if generate_dashboards:
        print(f"\nGenerating dashboards for top {top_k_features} features...")
        visualizer = FeatureVisualizer(sae, config)

        # Get top features
        top_indices, top_freqs = visualizer.get_top_features(activations, k=top_k_features)

        dashboards_dir = output_dir / "feature_dashboards"
        dashboards_dir.mkdir(exist_ok=True)

        for i, (idx, freq) in enumerate(zip(top_indices, top_freqs)):
            idx = idx.item()
            print(f"  Generating dashboard for feature {idx} (rank {i+1}, freq={freq:.4f})")

            dashboard_path = dashboards_dir / f"feature_{idx}.html"
            visualizer.save_feature_dashboard(
                feature_idx=idx,
                activations=activations,
                save_path=dashboard_path,
            )

        print(f"Saved dashboards to {dashboards_dir}")

    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAEs")

    # Input arguments
    parser.add_argument("--sae_path", type=str, default=None,
                       help="Path to SAE checkpoint")
    parser.add_argument("--sae_dir", type=str, default=None,
                       help="Directory containing multiple SAE checkpoints")

    # Evaluation arguments
    parser.add_argument("--num_test_samples", type=int, default=10000,
                       help="Number of test activations to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="sae_eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--generate_dashboards", action="store_true",
                       help="Generate feature dashboards")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top features to generate dashboards for")

    args = parser.parse_args()

    # Find SAE checkpoints to evaluate
    if args.sae_path:
        sae_paths = [Path(args.sae_path)]
    elif args.sae_dir:
        sae_dir = Path(args.sae_dir)
        # Find all best_model.pt or checkpoint files
        sae_paths = list(sae_dir.glob("**/best_model.pt"))
        if not sae_paths:
            sae_paths = list(sae_dir.glob("**/sae_final.pt"))
        if not sae_paths:
            sae_paths = list(sae_dir.glob("**/*.pt"))
    else:
        raise ValueError("Must specify either --sae_path or --sae_dir")

    if not sae_paths:
        print("No SAE checkpoints found!")
        return

    print(f"Found {len(sae_paths)} SAE checkpoint(s) to evaluate")

    # Evaluate each SAE
    all_results = []

    for sae_path in sae_paths:
        print(f"\n{'='*80}")
        print(f"Evaluating {sae_path}")
        print(f"{'='*80}")

        # Load SAE
        sae, config = load_sae_from_checkpoint(sae_path, device=args.device)

        # Generate test activations
        # In production, use real model activations
        print(f"Generating {args.num_test_samples} test activations...")
        test_activations = generate_test_activations(
            d_in=config.d_in,
            num_samples=args.num_test_samples,
            device=args.device,
        )

        # Create output directory for this SAE
        if args.sae_path:
            eval_output_dir = Path(args.output_dir)
        else:
            # Use relative path structure
            rel_path = sae_path.parent.relative_to(Path(args.sae_dir))
            eval_output_dir = Path(args.output_dir) / rel_path

        # Evaluate
        metrics, summary = evaluate_sae(
            sae=sae,
            config=config,
            activations=test_activations,
            output_dir=eval_output_dir,
            generate_dashboards=args.generate_dashboards,
            top_k_features=args.top_k,
        )

        all_results.append({
            "sae_path": str(sae_path),
            "hook_point": config.hook_point,
            "metrics": metrics.to_dict(),
            "summary": summary,
        })

    # Save combined results
    if len(all_results) > 1:
        combined_path = Path(args.output_dir) / "combined_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Saved combined results to {combined_path}")
        print(f"{'='*80}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
