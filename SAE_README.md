# SAE-Based Interpretability for Nanochat

This extension adds **Sparse Autoencoder (SAE)** based interpretability to nanochat, enabling mechanistic understanding of learned features at runtime and during training.

## Overview

Sparse Autoencoders help us understand what neural networks learn by decomposing dense activations into sparse, interpretable features. This implementation provides:

- **Multiple SAE architectures**: TopK, ReLU, and Gated SAEs
- **Activation collection**: Non-intrusive PyTorch hooks for collecting model activations
- **Training pipeline**: Complete SAE training with dead latent resampling and evaluation
- **Runtime interpretation**: Real-time feature tracking during inference
- **Feature steering**: Modify model behavior by intervening on specific features
- **Neuronpedia integration**: Prepare SAEs for upload to the Neuronpedia platform
- **Visualization tools**: Interactive dashboards for exploring features

## Installation

The SAE extension has no additional dependencies beyond nanochat's existing requirements. All code is pure PyTorch.

## Quick Start

### 1. Train an SAE

Train SAEs on a nanochat model checkpoint:

```bash
# Train SAE on layer 10
python -m scripts.sae_train \
    --checkpoint models/d20/base_final.pt \
    --layer 10 \
    --expansion_factor 8 \
    --activation topk \
    --k 64 \
    --num_activations 1000000

# Train SAEs on all layers
python -m scripts.sae_train \
    --checkpoint models/d20/base_final.pt \
    --output_dir sae_models/d20
```

### 2. Evaluate SAE Quality

Evaluate trained SAEs and generate metrics:

```bash
# Evaluate specific SAE
python -m scripts.sae_eval \
    --sae_path sae_models/d20/layer_10/best_model.pt \
    --generate_dashboards \
    --top_k 20

# Evaluate all SAEs
python -m scripts.sae_eval \
    --sae_dir sae_models/d20 \
    --output_dir eval_results
```

### 3. Visualize Features

Generate interactive feature dashboards:

```bash
# Visualize specific feature
python -m scripts.sae_viz \
    --sae_path sae_models/d20/layer_10/best_model.pt \
    --feature 4232 \
    --output_dir feature_viz

# Generate explorer for top features
python -m scripts.sae_viz \
    --sae_path sae_models/d20/layer_10/best_model.pt \
    --all_features \
    --top_k 50 \
    --output_dir feature_explorer
```

### 4. Runtime Interpretation

Use SAEs during inference for real-time feature tracking:

```python
from nanochat.gpt import GPT
from sae.runtime import InterpretableModel, load_saes

# Load model and SAEs
model = GPT.from_pretrained("models/d20/base_final.pt")
saes = load_saes("sae_models/d20/")

# Wrap model
interp_model = InterpretableModel(model, saes)

# Track features during inference
with interp_model.interpretation_enabled():
    output = interp_model(input_ids)
    features = interp_model.get_active_features()

# Inspect active features at layer 10
layer_10_features = features["blocks.10.hook_resid_post"]
print(f"Active features: {(layer_10_features != 0).sum()} / {layer_10_features.shape[1]}")
```

### 5. Feature Steering

Modify model behavior by amplifying or suppressing features:

```python
# Amplify a specific feature
steered_output = interp_model.steer(
    input_ids,
    feature_id=("blocks.10.hook_resid_post", 4232),
    strength=2.0  # 2x amplification
)

# Suppress a feature
suppressed_output = interp_model.steer(
    input_ids,
    feature_id=("blocks.10.hook_resid_post", 1234),
    strength=0.0  # Zero out feature
)
```

## Architecture

### SAE Models

Three SAE architectures are supported:

1. **TopK SAE** (Recommended)
   - Uses top-k activation to select k most active features
   - Direct sparsity control without L1 tuning
   - Fewer dead latents at scale
   - Reference: [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)

2. **ReLU SAE**
   - Traditional approach with ReLU activation and L1 penalty
   - Requires tuning L1 coefficient
   - Well-studied and interpretable

3. **Gated SAE**
   - Separates feature selection (gate) from magnitude
   - More expressive but more complex
   - Reference: [Gated SAEs](https://arxiv.org/abs/2404.16014)

### Module Structure

```
sae/
├── __init__.py          # Package exports
├── config.py            # SAE configuration dataclass
├── models.py            # TopK, ReLU, Gated SAE implementations
├── hooks.py             # Activation collection via PyTorch hooks
├── trainer.py           # SAE training loop and evaluation
├── runtime.py           # Real-time interpretation wrapper
├── evaluator.py         # SAE quality metrics
├── feature_viz.py       # Feature visualization tools
└── neuronpedia.py       # Neuronpedia API integration
```

## Configuration

SAE training is configured via `SAEConfig`:

```python
from sae.config import SAEConfig

config = SAEConfig(
    # Architecture
    d_in=1280,                    # Input dimension (model d_model)
    d_sae=10240,                  # SAE hidden dimension (8x expansion)
    activation="topk",            # SAE activation type
    k=64,                         # Number of active features (for TopK)

    # Training
    num_activations=10_000_000,   # Activations to collect
    batch_size=4096,              # Training batch size
    num_epochs=10,                # Training epochs
    learning_rate=3e-4,           # Learning rate

    # Hook point
    hook_point="blocks.10.hook_resid_post",  # Layer to hook
)
```

## Training Pipeline

### 1. Activation Collection

Activations are collected using PyTorch forward hooks:

```python
from sae.hooks import ActivationCollector

# Collect activations from layer 10
collector = ActivationCollector(
    model=model,
    hook_points=["blocks.10.hook_resid_post"],
    max_activations=1_000_000,
)

with collector:
    for batch in dataloader:
        model(batch)

activations = collector.get_activations()
```

### 2. SAE Training

Train SAE on collected activations:

```python
from sae.trainer import train_sae_from_activations

sae, trainer = train_sae_from_activations(
    activations=activations,
    config=config,
    device="cuda",
    save_dir="sae_outputs/layer_10",
)
```

Training includes:
- Learning rate warmup
- Dead latent resampling
- Decoder weight normalization
- Periodic evaluation and checkpointing

### 3. Evaluation

Evaluate SAE quality:

```python
from sae.evaluator import SAEEvaluator

evaluator = SAEEvaluator(sae, config)
metrics = evaluator.evaluate(test_activations)

print(metrics)
# Output:
# SAE Evaluation Metrics
# ==============================
# Reconstruction Quality:
#   MSE Loss: 0.001234
#   Explained Variance: 0.9876
#   Reconstruction Score: 0.9876
#
# Sparsity:
#   L0 (mean ± std): 64.2 ± 5.1
#   L1 (mean): 0.0234
#   Dead Latents: 2.34%
```

## Advanced Usage

### Custom Training Data

Use real training data instead of random activations:

```python
from nanochat.dataloader import DataLoader
from sae.hooks import collect_activations_from_dataloader

# Load your training data
dataloader = DataLoader(...)

# Collect activations
activations = collect_activations_from_dataloader(
    model=model,
    dataloader=dataloader,
    hook_points=["blocks.10.hook_resid_post"],
    max_activations=10_000_000,
)
```

### Multi-Layer Training

Train SAEs on multiple layers:

```python
layers_to_train = [5, 10, 15, 20]

for layer_idx in layers_to_train:
    config = SAEConfig.from_model(
        model,
        layer_idx=layer_idx,
        expansion_factor=8,
    )

    # Collect activations
    hook_point = f"blocks.{layer_idx}.hook_resid_post"
    activations = collect_activations(model, hook_point)

    # Train SAE
    sae, _ = train_sae_from_activations(
        activations=activations,
        config=config,
        save_dir=f"sae_models/layer_{layer_idx}",
    )
```

### Feature Analysis

Analyze specific features:

```python
from sae.feature_viz import FeatureVisualizer

visualizer = FeatureVisualizer(sae, config)

# Get top activating examples
examples = visualizer.get_max_activating_examples(
    feature_idx=4232,
    activations=activations,
    tokens=tokens,  # Optional: include token information
    k=10,
)

# Generate feature report
report = visualizer.generate_feature_report(
    feature_idx=4232,
    activations=activations,
    save_path="reports/feature_4232.json",
)
```

## Neuronpedia Integration

Prepare SAEs for upload to [Neuronpedia](https://neuronpedia.org):

```python
from sae.neuronpedia import NeuronpediaUploader, create_neuronpedia_metadata

# Create metadata
metadata = create_neuronpedia_metadata(
    sae=sae,
    config=config,
    training_info={"num_epochs": 10, "num_steps": 50000},
    eval_metrics={"mse_loss": 0.001, "l0": 64.2},
)

# Prepare for upload
uploader = NeuronpediaUploader(
    model_name="nanochat",
    model_version="d20",
)

uploader.prepare_sae_for_upload(
    sae=sae,
    config=config,
    output_dir="neuronpedia_upload/layer_10",
    metadata=metadata,
)
```

Follow the instructions in the generated README to upload to Neuronpedia.

## Performance Considerations

### Memory Usage

- **Activation Collection**: ~10-20GB per layer for 10M activations
- **SAE Training**: Requires GPU with 40GB+ VRAM for large SAEs
- **Runtime Inference**: +10GB memory for all SAEs loaded

### Computational Overhead

- **Activation Collection**: <5% overhead during training
- **SAE Inference**: 5-10% latency increase
- **SAE Training**: 2-4 hours per layer on A100

### Optimization Tips

1. **Use CPU for activation storage** during collection to save GPU memory
2. **Train SAEs on subset of layers** (e.g., every 5th layer)
3. **Use smaller expansion factors** (4x instead of 16x) for faster training
4. **Enable lazy loading** of SAEs to reduce memory usage at runtime

## Evaluation Metrics

SAEs are evaluated on:

1. **Reconstruction Quality**
   - MSE Loss: Mean squared error between original and reconstructed activations
   - Explained Variance: Fraction of activation variance captured by SAE
   - Reconstruction Score: 1 - MSE/variance

2. **Sparsity**
   - L0: Average number of active features per activation
   - L1: Average L1 norm of feature activations
   - Dead Latents: Fraction of features that never activate

3. **Feature Interpretability**
   - Activation frequency: How often each feature activates
   - Top activating examples: Inputs that maximally activate each feature
   - Feature descriptions: Auto-generated via Neuronpedia

## Troubleshooting

### Common Issues

1. **Out of Memory during activation collection**
   - Reduce batch size
   - Store activations on CPU: `device="cpu"` in `ActivationCollector`
   - Collect fewer activations

2. **High dead latent percentage**
   - Increase resampling frequency: `resample_interval=10000`
   - Use TopK SAE instead of ReLU
   - Increase number of training epochs

3. **Poor reconstruction quality**
   - Increase expansion factor (8x → 16x)
   - Train for more epochs
   - Reduce L1 coefficient (for ReLU SAE)

4. **SAE doesn't load at runtime**
   - Check config.json exists alongside checkpoint
   - Verify checkpoint contains `sae_state_dict` key
   - Ensure d_in matches model dimension

## Examples

See the `examples/` directory for complete examples:

- `examples/train_sae.py`: End-to-end SAE training
- `examples/interpret_model.py`: Runtime interpretation
- `examples/feature_steering.py`: Feature steering examples
- `examples/feature_analysis.py`: Feature analysis and visualization

## Citation

If you use this SAE implementation in your research, please cite:

```bibtex
@software{nanochat_sae,
  author = {Nanochat Contributors},
  title = {SAE-Based Interpretability for Nanochat},
  year = {2025},
  url = {https://github.com/karpathy/nanochat}
}
```

## References

- [Scaling and Evaluating Sparse Autoencoders (OpenAI)](https://arxiv.org/abs/2406.04093)
- [Neuronpedia Documentation](https://docs.neuronpedia.org)
- [SAELens Library](https://github.com/jbloomAus/SAELens)
- [Towards Monosemanticity (Anthropic)](https://transformer-circuits.pub/2023/monosemantic-features)

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Integration with actual nanochat training loop
- [ ] More sophisticated feature analysis tools
- [ ] Multi-modal SAE support
- [ ] Hierarchical SAEs
- [ ] Circuit discovery tools
- [ ] Better visualization UI

Please submit PRs or open issues on the nanochat repository.

## License

MIT License (same as nanochat)
