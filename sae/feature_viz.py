"""
Feature visualization and analysis tools for SAEs.

Provides utilities to visualize and understand SAE features.
"""

import torch
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from sae.models import BaseSAE
from sae.config import SAEConfig


class FeatureVisualizer:
    """Visualizer for SAE features."""

    def __init__(
        self,
        sae: BaseSAE,
        config: SAEConfig,
        tokenizer=None,
    ):
        """Initialize feature visualizer.

        Args:
            sae: SAE model
            config: SAE configuration
            tokenizer: Optional tokenizer for decoding tokens
        """
        self.sae = sae
        self.config = config
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_top_features(
        self,
        activations: torch.Tensor,
        k: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k most frequently active features.

        Args:
            activations: Activations to analyze, shape (num_samples, d_in)
            k: Number of top features to return

        Returns:
            Tuple of (feature_indices, activation_frequencies)
        """
        device = next(self.sae.parameters()).device
        activations = activations.to(device)

        # Get feature activations
        _, features, _ = self.sae(activations)

        # Count activation frequency for each feature
        activation_freq = (features != 0).float().mean(dim=0)

        # Get top-k
        topk_freq, topk_indices = torch.topk(activation_freq, k=min(k, len(activation_freq)))

        return topk_indices, topk_freq

    @torch.no_grad()
    def get_feature_statistics(
        self,
        feature_idx: int,
        activations: torch.Tensor,
    ) -> Dict[str, float]:
        """Get statistics for a specific feature.

        Args:
            feature_idx: Feature index
            activations: Activations to analyze

        Returns:
            Dictionary of statistics
        """
        device = next(self.sae.parameters()).device
        activations = activations.to(device)

        # Get feature activations
        _, features, _ = self.sae(activations)
        feature_acts = features[:, feature_idx]

        # Compute statistics
        activation_freq = (feature_acts != 0).float().mean().item()
        mean_activation = feature_acts.mean().item()
        max_activation = feature_acts.max().item()
        std_activation = feature_acts.std().item()

        non_zero = feature_acts[feature_acts != 0]
        if len(non_zero) > 0:
            mean_when_active = non_zero.mean().item()
            percentile_75 = torch.quantile(non_zero, 0.75).item()
            percentile_95 = torch.quantile(non_zero, 0.95).item()
        else:
            mean_when_active = 0.0
            percentile_75 = 0.0
            percentile_95 = 0.0

        return {
            "feature_idx": feature_idx,
            "activation_frequency": activation_freq,
            "mean_activation": mean_activation,
            "mean_when_active": mean_when_active,
            "max_activation": max_activation,
            "std_activation": std_activation,
            "percentile_75": percentile_75,
            "percentile_95": percentile_95,
        }

    @torch.no_grad()
    def get_max_activating_examples(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        k: int = 10,
    ) -> List[Dict]:
        """Get examples that maximally activate a feature.

        Args:
            feature_idx: Feature index
            activations: Activations, shape (num_samples, d_in)
            tokens: Optional token IDs corresponding to activations
            k: Number of examples to return

        Returns:
            List of dictionaries with activation info
        """
        device = next(self.sae.parameters()).device
        activations = activations.to(device)

        # Get feature activations
        _, features, _ = self.sae(activations)
        feature_acts = features[:, feature_idx]

        # Get top-k activating examples
        topk_acts, topk_indices = torch.topk(feature_acts, k=min(k, len(feature_acts)))

        examples = []
        for i, (idx, act) in enumerate(zip(topk_indices, topk_acts)):
            idx = idx.item()
            act = act.item()

            example = {
                "rank": i + 1,
                "activation": act,
                "sample_idx": idx,
            }

            # Add token info if available
            if tokens is not None and self.tokenizer is not None:
                token_id = tokens[idx].item()
                token_str = self.tokenizer.decode([token_id])
                example["token_id"] = token_id
                example["token_str"] = token_str

            examples.append(example)

        return examples

    def generate_feature_report(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Generate comprehensive report for a feature.

        Args:
            feature_idx: Feature index
            activations: Activations to analyze
            tokens: Optional tokens
            save_path: Optional path to save report

        Returns:
            Dictionary with feature report
        """
        stats = self.get_feature_statistics(feature_idx, activations)
        examples = self.get_max_activating_examples(
            feature_idx, activations, tokens=tokens, k=20
        )

        report = {
            "feature_idx": feature_idx,
            "hook_point": self.config.hook_point,
            "statistics": stats,
            "max_activating_examples": examples,
        }

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Saved feature report to {save_path}")

        return report

    def visualize_feature_dashboard(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate HTML dashboard for a feature.

        Args:
            feature_idx: Feature index
            activations: Activations
            tokens: Optional tokens

        Returns:
            HTML string
        """
        report = self.generate_feature_report(feature_idx, activations, tokens)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature {feature_idx} Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ margin: 20px 0; }}
                .stat-item {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .stat-label {{ font-weight: bold; }}
                .examples {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Feature {feature_idx}</h1>
                <p>Hook Point: {report['hook_point']}</p>
            </div>

            <div class="stats">
                <h2>Statistics</h2>
        """

        stats = report['statistics']
        for key, value in stats.items():
            if key != "feature_idx":
                html += f'<div class="stat-item"><span class="stat-label">{key}:</span> {value:.4f}</div>'

        html += """
            </div>

            <div class="examples">
                <h2>Top Activating Examples</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Activation</th>
                        <th>Sample Index</th>
        """

        if tokens is not None:
            html += "<th>Token</th>"

        html += "</tr>"

        for ex in report['max_activating_examples'][:10]:
            html += f"""
                <tr>
                    <td>{ex['rank']}</td>
                    <td>{ex['activation']:.4f}</td>
                    <td>{ex['sample_idx']}</td>
            """
            if 'token_str' in ex:
                html += f"<td>{ex['token_str']}</td>"
            html += "</tr>"

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        return html

    def save_feature_dashboard(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        save_path: Path,
        tokens: Optional[torch.Tensor] = None,
    ):
        """Save feature dashboard as HTML.

        Args:
            feature_idx: Feature index
            activations: Activations
            save_path: Path to save HTML
            tokens: Optional tokens
        """
        html = self.visualize_feature_dashboard(feature_idx, activations, tokens)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write(html)

        print(f"Saved feature dashboard to {save_path}")


def generate_sae_summary(
    sae: BaseSAE,
    config: SAEConfig,
    activations: torch.Tensor,
    save_path: Optional[Path] = None,
) -> Dict:
    """Generate summary report for an entire SAE.

    Args:
        sae: SAE model
        config: SAE configuration
        activations: Sample activations for analysis
        save_path: Optional path to save summary

    Returns:
        Dictionary with SAE summary
    """
    visualizer = FeatureVisualizer(sae, config)

    # Get top features by activation frequency
    top_indices, top_freqs = visualizer.get_top_features(activations, k=100)

    # Get statistics for top features
    top_features_info = []
    for idx, freq in zip(top_indices[:20], top_freqs[:20]):
        idx = idx.item()
        freq = freq.item()
        stats = visualizer.get_feature_statistics(idx, activations)
        top_features_info.append({
            "feature_idx": idx,
            "activation_frequency": freq,
            "mean_when_active": stats["mean_when_active"],
        })

    summary = {
        "hook_point": config.hook_point,
        "d_in": config.d_in,
        "d_sae": config.d_sae,
        "activation_type": config.activation,
        "expansion_factor": config.d_sae / config.d_in,
        "top_features": top_features_info,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved SAE summary to {save_path}")

    return summary
