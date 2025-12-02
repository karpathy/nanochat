#!/bin/bash
set -e

PROJECT="nzp-nanochat"
NETWORK_NAME="nanochat-network"

echo "Setting up network resources for project $PROJECT..."

# 1. Create the VPC network (auto mode creates subnets in all regions)
if ! gcloud compute networks describe "$NETWORK_NAME" --project="$PROJECT" &>/dev/null; then
    echo "Creating VPC network '$NETWORK_NAME'..."
    gcloud compute networks create "$NETWORK_NAME" \
        --project="$PROJECT" \
        --subnet-mode=auto \
        --bgp-routing-mode=global
    echo "✅ Network created."
else
    echo "✅ Network '$NETWORK_NAME' already exists."
fi

# 2. Create firewall rule to allow internal communication
if ! gcloud compute firewall-rules describe "${NETWORK_NAME}-allow-internal" --project="$PROJECT" &>/dev/null; then
    echo "Creating firewall rule '${NETWORK_NAME}-allow-internal'..."
    gcloud compute firewall-rules create "${NETWORK_NAME}-allow-internal" \
        --project="$PROJECT" \
        --network="$NETWORK_NAME" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.128.0.0/9
    echo "✅ Firewall rule created."
else
    echo "✅ Firewall rule '${NETWORK_NAME}-allow-internal' already exists."
fi

echo "Network setup complete!"
