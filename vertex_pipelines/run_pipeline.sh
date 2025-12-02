#!/bin/bash
set -euo pipefail

# Check for optional flags
SKIP_BUILD=false
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=true
    shift
fi

# Check for required arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 10 ]; then
    echo "Usage: $0 [--skip-build] gs://YOUR_GCS_BUCKET [WANDB_RUN_ID] [VERTEX_EXPERIMENT] [VERTEX_TENSORBOARD] [REGION] [ACCELERATOR_TYPE] [ACCELERATOR_COUNT] [NUM_DATA_SHARDS] [PREEMPTIBLE] [DEVICE_BATCH_SIZE]"
    echo "  REGION defaults to \$VERTEX_REGION env var or 'us-central1'"
    echo "  ACCELERATOR_TYPE defaults to 'NVIDIA_L4'"
    echo "  ACCELERATOR_COUNT defaults to 1"
    echo "  NUM_DATA_SHARDS defaults to 20 (number of HuggingFace data shards to download)"
    echo "  PREEMPTIBLE defaults to false"
    echo "  DEVICE_BATCH_SIZE defaults to 8"
    exit 1
fi

if [[ ! "$1" =~ ^gs:// ]]; then
  echo "Error: GCS bucket must be a valid gs:// path."
  echo "Usage: $0 [--skip-build] gs://YOUR_GCS_BUCKET [WANDB_RUN_ID] [VERTEX_EXPERIMENT] [VERTEX_TENSORBOARD] [REGION] [ACCELERATOR_TYPE] [ACCELERATOR_COUNT] [NUM_DATA_SHARDS] [PREEMPTIBLE] [DEVICE_BATCH_SIZE]"
  exit 1
fi

GCS_BUCKET=$1
PIPELINE_ROOT="$GCS_BUCKET/pipeline-root"
GCP_PROJECT=$(gcloud config get-value project)
WANDB_RUN=${2:-"dummy"} # Default to "dummy" if not provided
VERTEX_EXPERIMENT=${3:-""}
VERTEX_TENSORBOARD=${4:-""}
REGION=${5:-${VERTEX_REGION:-us-central1}} # Use arg, then env var, then default
ACCELERATOR_TYPE=${6:-NVIDIA_L4}
ACCELERATOR_COUNT=${7:-1}
NUM_DATA_SHARDS=${8:-20}
PREEMPTIBLE=${9:-false}
DEVICE_BATCH_SIZE=${10:-8}

echo "Using GCP Project: $GCP_PROJECT"
echo "Using GCS Bucket: $GCS_BUCKET"
echo "Using Region: $REGION"
echo "Using Accelerator: $ACCELERATOR_TYPE"
echo "Using WANDB Run ID: $WANDB_RUN"
if [ -n "$VERTEX_EXPERIMENT" ]; then
    echo "Using Vertex Experiment: $VERTEX_EXPERIMENT"
fi
if [ -n "$VERTEX_TENSORBOARD" ]; then
    echo "Using Vertex TensorBoard: $VERTEX_TENSORBOARD"
fi

# Submit the build to Cloud Build and get the image URI with digest
# Use a timestamp tag to avoid caching issues with 'latest'
if [ -z "${DOCKER_IMAGE_URI:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    IMAGE_URI="gcr.io/$GCP_PROJECT/nanochat:$TIMESTAMP"
else
    TIMESTAMP="custom"
    IMAGE_URI="$DOCKER_IMAGE_URI"
fi
if [ "$SKIP_BUILD" = false ]; then
    echo "Submitting build to Cloud Build with tag $TIMESTAMP..."
    gcloud builds submit --config vertex_pipelines/cloudbuild.yaml --substitutions=_IMAGE_NAME="$IMAGE_URI" . --project=$GCP_PROJECT
    echo "Cloud Build completed."
else
    echo "Skipping Cloud Build."
fi
echo "Using image URI: $IMAGE_URI"

# Run the Vertex AI pipeline
# Install dependencies for pipeline compilation
echo "Installing dependencies..."
if [ ! -d ".venv_pipeline" ]; then
    python3 -m venv .venv_pipeline
fi
source .venv_pipeline/bin/activate
python3 -m pip install -r requirements.txt

echo "Running Vertex AI pipeline..."
export DOCKER_IMAGE_URI="$IMAGE_URI"
# Use the default compute service account for the project
SERVICE_ACCOUNT="247010501180-compute@developer.gserviceaccount.com"

python3 vertex_pipelines/pipeline.py \
    --gcp-project "$GCP_PROJECT" \
    --gcs-bucket "$GCS_BUCKET" \
    --pipeline-root "$PIPELINE_ROOT" \
    --region "$REGION" \
    --wandb-run "$WANDB_RUN" \
    --vertex-experiment "$VERTEX_EXPERIMENT" \
    --vertex-tensorboard "$VERTEX_TENSORBOARD" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --accelerator-count "$ACCELERATOR_COUNT" \
    --preemptible "$PREEMPTIBLE" \
    --num-data-shards "$NUM_DATA_SHARDS" \
    --service-account "$SERVICE_ACCOUNT" \
    --device-batch-size "$DEVICE_BATCH_SIZE"

echo "Pipeline submitted."