#!/bin/bash
set -euo pipefail

# Usage: ./setup_resources.sh <PROJECT_ID> <REGION> <BUCKET_NAME> [EXPERIMENT_NAME] [TENSORBOARD_DISPLAY_NAME]

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <PROJECT_ID> <REGION> <BUCKET_NAME> [EXPERIMENT_NAME] [TENSORBOARD_DISPLAY_NAME]"
    exit 1
fi

PROJECT_ID=$1
REGION=$2
BUCKET_NAME=$3
EXPERIMENT_NAME=${4:-"nanochat-experiment"}
TENSORBOARD_DISPLAY_NAME=${5:-"nanochat-tensorboard"}

echo "Setting up resources in Project: $PROJECT_ID, Region: $REGION"

# 1. Create GCS Bucket
echo "Checking bucket gs://$BUCKET_NAME..."
if gcloud storage buckets describe "gs://$BUCKET_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "Bucket gs://$BUCKET_NAME already exists."
else
    echo "Creating bucket gs://$BUCKET_NAME..."
    gcloud storage buckets create "gs://$BUCKET_NAME" --project="$PROJECT_ID" --location="$REGION" --uniform-bucket-level-access
    echo "Bucket created."
fi

# 2. Create Vertex AI TensorBoard
echo "Checking for existing TensorBoard with display name: $TENSORBOARD_DISPLAY_NAME..."
EXISTING_TB=$(gcloud ai tensorboards list --region="$REGION" --project="$PROJECT_ID" --filter="displayName=$TENSORBOARD_DISPLAY_NAME" --format="value(name)" 2>/dev/null || true)

if [ -n "$EXISTING_TB" ]; then
    echo "TensorBoard '$TENSORBOARD_DISPLAY_NAME' already exists: $EXISTING_TB"
    TENSORBOARD_ID=$EXISTING_TB
else
    echo "Creating Vertex AI TensorBoard: $TENSORBOARD_DISPLAY_NAME..."
    # Create and capture the output. The output usually contains the name.
    # We use --format="value(name)" to get just the resource name.
    TENSORBOARD_ID=$(gcloud ai tensorboards create --display-name="$TENSORBOARD_DISPLAY_NAME" --region="$REGION" --project="$PROJECT_ID" --format="value(name)")
    echo "TensorBoard created: $TENSORBOARD_ID"
fi

# 3. Create Vertex AI Experiment
echo "Creating Vertex AI Experiment: $EXPERIMENT_NAME..."
# Experiments are often implicitly created, but we can explicitly create it.
# We check if it exists first to avoid errors.
if gcloud ai experiments list --region="$REGION" --project="$PROJECT_ID" --filter="name=$EXPERIMENT_NAME" --format="value(name)" 2>/dev/null | grep -q "$EXPERIMENT_NAME"; then
     echo "Experiment '$EXPERIMENT_NAME' already exists."
else
     # Try to create. Note: 'gcloud ai experiments create' might fail if it already exists but wasn't found by list for some reason,
     # or if the command syntax varies. We'll allow it to fail gracefully if it's just "already exists".
     gcloud ai experiments create --experiment="$EXPERIMENT_NAME" --region="$REGION" --project="$PROJECT_ID" || echo "Experiment creation returned status $? (might already exist)."
     echo "Experiment setup complete."
fi

echo "----------------------------------------------------------------"
echo "Setup Complete!"
echo "----------------------------------------------------------------"
echo "Use the following values for run_pipeline.sh:"
echo ""
echo "GCS_BUCKET: gs://$BUCKET_NAME"
echo "VERTEX_EXPERIMENT: $EXPERIMENT_NAME"
echo "VERTEX_TENSORBOARD: $TENSORBOARD_ID"
echo ""
echo "Example Command:"
echo "./vertex_pipelines/run_pipeline.sh gs://$BUCKET_NAME <WANDB_RUN> $EXPERIMENT_NAME $TENSORBOARD_ID"
echo "----------------------------------------------------------------"
