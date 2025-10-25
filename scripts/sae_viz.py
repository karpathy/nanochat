"""
Visualize and explore SAE features interactively.

This script provides interactive exploration of trained SAEs.

Usage:
    # Explore specific feature
    python -m scripts.sae_viz --sae_path sae_outputs/layer_10/best_model.pt \
        --feature 4232

    # Generate all dashboards
    python -m scripts.sae_viz --sae_path sae_outputs/layer_10/best_model.pt \
        --all_features --output_dir dashboards
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
from sae.feature_viz import FeatureVisualizer


def load_sae_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load SAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    if "config" in checkpoint:
        config = SAEConfig.from_dict(checkpoint["config"])
    else:
        config_path = checkpoint_path.parent / "config.json"
        with open(config_path) as f:
            config = SAEConfig.from_dict(json.load(f))

    # Create and load SAE
    sae = create_sae(config)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    sae.to(device)
    sae.eval()

    return sae, config


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE features")

    # Input arguments
    parser.add_argument("--sae_path", type=str, required=True,
                       help="Path to SAE checkpoint")
    parser.add_argument("--feature", type=int, default=None,
                       help="Specific feature index to visualize")
    parser.add_argument("--all_features", action="store_true",
                       help="Generate dashboards for all features")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Number of top features to visualize if --all_features")

    # Data arguments
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of activation samples to use")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="feature_viz",
                       help="Directory to save visualizations")

    args = parser.parse_args()

    # Load SAE
    print(f"Loading SAE from {args.sae_path}")
    sae, config = load_sae_from_checkpoint(Path(args.sae_path), device=args.device)

    # Generate test activations
    print(f"Generating {args.num_samples} test activations...")
    test_activations = torch.randn(args.num_samples, config.d_in, device=args.device)

    # Create visualizer
    visualizer = FeatureVisualizer(sae, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature is not None:
        # Visualize specific feature
        print(f"\nVisualizing feature {args.feature}")

        # Get statistics
        stats = visualizer.get_feature_statistics(args.feature, test_activations)
        print("\nFeature Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")

        # Generate dashboard
        dashboard_path = output_dir / f"feature_{args.feature}.html"
        visualizer.save_feature_dashboard(
            feature_idx=args.feature,
            activations=test_activations,
            save_path=dashboard_path,
        )
        print(f"\nSaved dashboard to {dashboard_path}")

        # Generate report
        report_path = output_dir / f"feature_{args.feature}_report.json"
        report = visualizer.generate_feature_report(
            feature_idx=args.feature,
            activations=test_activations,
            save_path=report_path,
        )

    elif args.all_features:
        # Visualize top features
        print(f"\nFinding top {args.top_k} features...")
        top_indices, top_freqs = visualizer.get_top_features(test_activations, k=args.top_k)

        print(f"Generating dashboards for top {len(top_indices)} features...")
        for i, (idx, freq) in enumerate(zip(top_indices, top_freqs)):
            idx = idx.item()
            print(f"  [{i+1}/{len(top_indices)}] Feature {idx} (freq={freq:.4f})")

            dashboard_path = output_dir / f"feature_{idx}.html"
            visualizer.save_feature_dashboard(
                feature_idx=idx,
                activations=test_activations,
                save_path=dashboard_path,
            )

        # Create index page
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAE Feature Explorer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .feature-list { margin: 20px 0; }
                .feature-item {
                    padding: 10px;
                    margin: 5px 0;
                    background: #fff;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                }
                .feature-item:hover { background: #f9f9f9; }
                a { text-decoration: none; color: #4CAF50; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SAE Feature Explorer</h1>
                <p>Hook Point: """ + config.hook_point + """</p>
                <p>Total Features: """ + str(config.d_sae) + """</p>
            </div>
            <div class="feature-list">
                <h2>Top Features</h2>
        """

        for i, (idx, freq) in enumerate(zip(top_indices, top_freqs)):
            idx = idx.item()
            index_html += f"""
                <div class="feature-item">
                    <a href="feature_{idx}.html">
                        Feature {idx} - Activation Frequency: {freq:.4f}
                    </a>
                </div>
            """

        index_html += """
            </div>
        </body>
        </html>
        """

        index_path = output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write(index_html)

        print(f"\nSaved feature explorer to {index_path}")
        print(f"Open in browser: file://{index_path.absolute()}")

    else:
        print("Please specify either --feature or --all_features")
        return

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
