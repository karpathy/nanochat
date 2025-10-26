# Modal Deployment Guide

This guide explains how to deploy and run nanochat on Modal's serverless infrastructure.

## Quick Start

1. **Install and authenticate with Modal:**
   ```bash
   uv run modal setup
   ```

2. **Choose your workflow:**

## Files Overview

### `modal_smoke.py` - Quick Image Setup & Testing

This script sets up the Modal image and runs quick smoke tests to verify everything works. Use this to:
- Build the Modal image with all dependencies
- Test that the environment is correctly configured
- Quickly validate your setup before running longer training jobs

```bash
uv run modal run modal_smoke.py
```

### `modal_speedrun.py` - Full Training Pipeline

This script runs the complete nanochat training pipeline on Modal. You can:
- Run the entire pipeline (base training → midtraining → SFT)
- Run only SFT if you already have a base model
- Train on 8xB200 GPUs (completes in under 2 hours)

```bash
# Full training pipeline
uv run modal run modal_speedrun.py

# SFT only (requires existing base model)
uv run modal run modal_speedrun.py --mode sft
```

### `modal_serve.py` - Interactive Chat Interface

This script deploys the web UI so you can chat with your trained model.

```bash
# Development mode (stays running while terminal is open)
uv run modal serve modal_serve.py

# Production deployment (runs independently)
uv run modal deploy modal_serve.py
```

Modal will print a URL - visit it in your browser to chat with your model!

## Model Loading Behavior

**Important:** Under the hood, the `load_model()` function automatically picks:
- The **highest model variant** available
- The **checkpoint with the most steps**

To override this behavior, explicitly specify the model tag and step in the `sys.argv` configuration of the Modal script you're using.

## Volume Structure

All scripts use the same `nanochat-data` volume with this structure:

```
/data/.cache/nanochat/
├── chatsft_checkpoints/     # SFT checkpoints
├── mid_checkpoints/         # Midtraining checkpoints
├── base_checkpoints/        # Base model checkpoints
└── tokenizer/               # Trained tokenizer
```

## Monitoring & Debugging

```bash
# View logs
modal app logs nanochat-serve

# Check volume contents
modal volume ls nanochat-data /.cache/nanochat

# Download checkpoints
modal volume get nanochat-data /.cache/nanochat/chatsft_checkpoints ./checkpoints
```

Visit the Modal dashboard at https://modal.com/apps for more details.
