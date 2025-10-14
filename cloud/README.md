# Running nanochat on the Cloud with SkyPilot

This directory contains [SkyPilot](https://skypilot.readthedocs.io/) configurations for easily launching nanochat on various cloud providers.

## Prerequisites

1. Install SkyPilot and configure it with your cloud provider(s):
   - Follow the [SkyPilot installation guide](https://docs.skypilot.co/en/latest/getting-started/installation.html)
   - Configure your cloud credentials (AWS, GCP, Azure, Lambda, Nebius, etc.)

## Training: Running the Speedrun Pipeline

Launch the speedrun training pipeline on any cloud provider with a single command:

```bash
sky launch -c nanochat-speedrun cloud/speedrun.sky.yaml --infra <aws|gcp|nebius|lambda|etc>
```

This will:
- Provision an 8xH100 GPU node
- Set up the environment
- Run the complete training pipeline via `speedrun.sh`
- Save trained model checkpoints to `s3://nanochat-data` (change this to your own bucket)
- Complete in approximately 4 hours (~$100 on most providers)

### Monitoring Training Progress

After launching, you can SSH into the cluster and monitor progress:

```bash
# SSH into the cluster
ssh nanochat-speedrun

# View the speedrun logs
sky logs nanochat-speedrun
```

## Serving: Deploy Your Trained Model

Once training is complete, serve your trained model with the web UI:

```bash
sky launch -c nanochat-serve cloud/serve.sky.yaml --infra <aws|gcp|nebius|lambda|etc>
```

This will:
- Provision a 1xH100 GPU node (much cheaper then an 8xH100 VM used for training)
- Load model weights from the same `s3://nanochat-data` bucket used during training
- Serve the web chat interface on port 8000
- Cost is ~$2-3/hour on most providers

### Accessing the Web UI

Get the endpoint URL to access the chat interface:

```bash
sky status --endpoint 8000 nanochat-serve
```

Open the displayed URL in your browser to chat with your trained model!

### Shared Storage

Both training and serving tasks use [SkyPilot's bucket mounting functionality](https://docs.skypilot.co/en/latest/reference/storage.html) to preserve and share model weights. This allows you to:
- Train once, serve multiple times without re-downloading weights
- Share trained models across different serving instances


