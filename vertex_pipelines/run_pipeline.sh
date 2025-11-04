#!/bin/bash
set -euo pipefail

# Check for required arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 gs://YOUR_GCS_BUCKET"
    exit 1
fi

if [[ ! "$1" =~ ^gs:// ]]; then
  echo "Error: GCS bucket must be a valid gs:// path."
  echo "Usage: $0 gs://YOUR_GCS_BUCKET"
  exit 1
fi

GCS_BUCKET=$1
PIPELINE_ROOT="$GCS_BUCKET/pipeline-root"
GCP_PROJECT=$(gcloud config get-value project)
REGION="us-central1"

echo "Using GCP Project: $GCP_PROJECT"
echo "Using GCS Bucket: $GCS_BUCKET"
echo "Using Region: $REGION"

# Submit the build to Cloud Build and get the image URI with digest
echo "Submitting build to Cloud Build..."
IMAGE_URI=$(gcloud builds submit --config vertex_pipelines/cloudbuild.yaml --format="value(results.images[0].name)" . --project=$GCP_PROJECT)
echo "Cloud Build completed. Using image URI: $IMAGE_URI"

# Run the Vertex AI pipeline
echo "Running Vertex AI pipeline..."
python vertex_pipelines/pipeline.py \
    --gcp-project "$GCP_PROJECT" \
    --gcs-bucket "$GCS_BUCKET" \
    --pipeline-root "$PIPELINE_ROOT" \
    --docker-image-uri "$IMAGE_URI" \
    --region "$REGION"

echo "Pipeline submitted."
