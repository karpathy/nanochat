"""
Train Sparse Autoencoders on nanochat activations.

This script trains SAEs on collected activations from a nanochat model.

Usage:
    # Train SAEs on all layers
    python -m scripts.sae_train --checkpoint models/d20/base_final.pt

    # Train SAE on specific layer
    python -m scripts.sae_train --checkpoint models/d20/base_final.pt --layer 10

    # Custom configuration
    python -m scripts.sae_train --checkpoint models/d20/base_final.pt \
        --layer 10 --expansion_factor 16 --activation topk --k 128
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_dist_info
from sae.config import SAEConfig
from sae.hooks import ActivationCollector
from sae.trainer import train_sae_from_activations
from sae.runtime import save_sae
from sae.neuronpedia import NeuronpediaUploader, create_neuronpedia_metadata


def load_model(checkpoint_path: Path, device: str = "cuda"):
    """Load nanochat model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config_dict = checkpoint.get("config", {})

    # Create GPT config
    config = GPTConfig(
        sequence_len=config_dict.get("sequence_len", 1024),
        vocab_size=config_dict.get("vocab_size", 50304),
        n_layer=config_dict.get("n_layer", 20),
        n_head=config_dict.get("n_head", 10),
        n_kv_head=config_dict.get("n_kv_head", 10),
        n_embd=config_dict.get("n_embd", 1280),
    )

    # Create model
    model = GPT(config)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    return model, config


def collect_activations_simple(
    model: GPT,
    hook_point: str,
    num_activations: int = 1_000_000,
    device: str = "cuda",
    sequence_length: int = 1024,
    batch_size: int = 8,
):
    """Collect activations using random data (for demonstration).

    In production, you would use actual training data.
    """
    print(f"Collecting {num_activations} activations from {hook_point}")

    collector = ActivationCollector(
        model=model,
        hook_points=[hook_point],
        max_activations=num_activations,
        device="cpu",  # Store on CPU to save GPU memory
    )

    model.eval()
    with torch.no_grad(), collector:
        num_samples_needed = (num_activations // (sequence_length * batch_size)) + 1

        for i in range(num_samples_needed):
            # Generate random tokens (in production, use real data)
            tokens = torch.randint(
                0,
                model.config.vocab_size,
                (batch_size, sequence_length),
                device=device
            )

            # Forward pass
            _ = model(tokens)

            # Check if we have enough
            if collector.counts[hook_point] >= num_activations:
                break

            if (i + 1) % 10 == 0:
                print(f"  Collected {collector.counts[hook_point]:,} activations...")

    activations = collector.get_activations()[hook_point]
    print(f"Collected {activations.shape[0]:,} activations, shape: {activations.shape}")

    return activations


def main():
    parser = argparse.ArgumentParser(description="Train SAEs on nanochat activations")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to nanochat checkpoint")
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer to train SAE on (if None, trains on all layers)")
    parser.add_argument("--hook_type", type=str, default="resid_post",
                       choices=["resid_post", "attn", "mlp"],
                       help="Type of hook point")

    # SAE architecture arguments
    parser.add_argument("--expansion_factor", type=int, default=8,
                       help="SAE expansion factor (d_sae = d_in * expansion_factor)")
    parser.add_argument("--activation", type=str, default="topk",
                       choices=["topk", "relu", "gated"],
                       help="SAE activation function")
    parser.add_argument("--k", type=int, default=64,
                       help="Number of active features for TopK SAE")
    parser.add_argument("--l1_coefficient", type=float, default=1e-3,
                       help="L1 coefficient for ReLU SAE")

    # Training arguments
    parser.add_argument("--num_activations", type=int, default=1_000_000,
                       help="Number of activations to collect for training")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="sae_outputs",
                       help="Directory to save trained SAEs")
    parser.add_argument("--prepare_neuronpedia", action="store_true",
                       help="Prepare SAE for Neuronpedia upload")

    args = parser.parse_args()

    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, model_config = load_model(checkpoint_path, device=args.device)

    # Determine layers to train
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = range(model_config.n_layer)

    # Train SAE for each layer
    for layer_idx in layers:
        print(f"\n{'='*80}")
        print(f"Training SAE for layer {layer_idx}")
        print(f"{'='*80}")

        hook_point = f"blocks.{layer_idx}.hook_{args.hook_type}"

        # Create SAE config
        sae_config = SAEConfig(
            d_in=model_config.n_embd,
            hook_point=hook_point,
            expansion_factor=args.expansion_factor,
            activation=args.activation,
            k=args.k,
            l1_coefficient=args.l1_coefficient,
            num_activations=args.num_activations,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )

        print(f"SAE Config:")
        print(f"  d_in: {sae_config.d_in}")
        print(f"  d_sae: {sae_config.d_sae}")
        print(f"  activation: {sae_config.activation}")
        print(f"  expansion_factor: {sae_config.expansion_factor}x")

        # Collect activations
        activations = collect_activations_simple(
            model=model,
            hook_point=hook_point,
            num_activations=args.num_activations,
            device=args.device,
        )

        # Create output directory
        output_dir = Path(args.output_dir) / f"layer_{layer_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train SAE
        print(f"\nTraining SAE...")
        sae, trainer = train_sae_from_activations(
            activations=activations,
            config=sae_config,
            device=args.device,
            save_dir=output_dir,
            verbose=True,
        )

        # Save final SAE
        save_path = output_dir / "sae_final.pt"
        save_sae(
            sae=sae,
            config=sae_config,
            save_path=save_path,
            training_steps=trainer.step,
            best_val_loss=trainer.best_val_loss,
        )

        print(f"\nSaved SAE to {save_path}")

        # Prepare for Neuronpedia upload if requested
        if args.prepare_neuronpedia:
            print(f"\nPreparing SAE for Neuronpedia upload...")

            # Determine model version from checkpoint path
            model_version = "d20"  # Default
            if "d26" in str(checkpoint_path):
                model_version = "d26"
            elif "d30" in str(checkpoint_path):
                model_version = "d30"

            uploader = NeuronpediaUploader(
                model_name="nanochat",
                model_version=model_version,
            )

            # Get evaluation metrics
            eval_metrics = {}
            if trainer.val_losses:
                last_val = trainer.val_losses[-1]
                eval_metrics = {
                    "mse_loss": last_val.get("mse_loss", 0),
                    "l0": last_val.get("l0", 0),
                }

            metadata = create_neuronpedia_metadata(
                sae=sae,
                config=sae_config,
                training_info={
                    "num_epochs": args.num_epochs,
                    "num_steps": trainer.step,
                    "num_activations": args.num_activations,
                },
                eval_metrics=eval_metrics,
            )

            neuronpedia_dir = output_dir / "neuronpedia_upload"
            uploader.prepare_sae_for_upload(
                sae=sae,
                config=sae_config,
                output_dir=neuronpedia_dir,
                metadata=metadata,
            )

    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"SAEs saved to {Path(args.output_dir)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
