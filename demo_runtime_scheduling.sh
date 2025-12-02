#!/bin/bash
# Demonstration: Submitting the same compiled pipeline with different scheduling strategies
# without recompilation

set -e

echo "=== Demo: Runtime Scheduling Strategy Changes ==="
echo ""
echo "This demonstrates that we can now change scheduling strategies"
echo "without recompiling the pipeline or rebuilding the Docker image."
echo ""

# Compile the pipeline once
echo "1. Compiling pipeline (one time)..."
python3 vertex_pipelines/pipeline.py \
    --gcp-project nzp-nanochat \
    --gcs-bucket gs://nzp-nanochat \
    --pipeline-root gs://nzp-nanochat/pipeline-root \
    --region us-central1 \
    --wandb-run test-run \
    --vertex-experiment nanochat-experiment \
    --vertex-tensorboard projects/247010501180/locations/us-central1/tensorboards/8180826106513850368 \
    --accelerator-type NVIDIA_TESLA_A100 \
    --accelerator-count 8 \
    --preemptible true \
    --num-data-shards 20 \
    --service-account 247010501180-compute@developer.gserviceaccount.com \
    --template_path demo_pipeline.json \
    2>&1 | grep -v "^Creating\|^To use\|^View\|state:"

echo "✓ Pipeline compiled successfully"
echo ""

# Show the scheduling parameters in the compiled pipeline
echo "2. Checking compiled pipeline parameters..."
python3 -c "
import json
data = json.load(open('demo_pipeline.json'))
params = data['root']['inputDefinitions']['parameters']
print('   scheduling_strategy: default =', params['scheduling_strategy']['defaultValue'])
print('   max_wait_duration: default =', params['max_wait_duration']['defaultValue'])
"
echo ""

echo "3. Demonstrating runtime parameter override..."
echo "   You can now submit this compiled pipeline with different strategies:"
echo ""
echo "   Option A (DWS - wait indefinitely):"
echo "   --scheduling-strategy FLEX_START --max-wait-duration 0s"
echo ""
echo "   Option B (DWS - wait 1 hour):"
echo "   --scheduling-strategy FLEX_START --max-wait-duration 3600s"
echo ""
echo "   Option C (Standard on-demand):"
echo "   --scheduling-strategy STANDARD --max-wait-duration 86400s"
echo ""
echo "   Option D (Legacy Spot):"
echo "   --scheduling-strategy SPOT --max-wait-duration 0s"
echo ""

echo "=== Summary ==="
echo "✓ Pipeline compilation is DECOUPLED from scheduling configuration"
echo "✓ No recompilation needed when changing FLEX_START ↔ SPOT ↔ STANDARD"
echo "✓ No Docker rebuild needed for deployment strategy changes"
echo ""
echo "To submit with a different strategy, just pass:"
echo "  --scheduling-strategy <VALUE> --max-wait-duration <VALUE>"
echo "to pipeline.py or add them to run_pipeline.sh"
