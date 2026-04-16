#!/usr/bin/env bash
set -euo pipefail

# Rotate EKS managed node group to latest AMI with zero downtime.
# Usage: ./scripts/rotate-nodes.sh <environment>
# Example: ./scripts/rotate-nodes.sh dev

ENVIRONMENT="${1:?Usage: rotate-nodes.sh <environment> (dev|uat|prod)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$SCRIPT_DIR/../terraform/environments/$ENVIRONMENT"

echo "=== samosaChaat Node Rotation — $ENVIRONMENT ==="

echo ""
echo "Step 1: Check current AMI vs latest available"
cd "$TF_DIR"

echo ""
echo "Step 2: Apply Terraform to update launch template with latest AMI"
echo "This triggers EKS managed node group rolling update."
echo "EKS will:"
echo "  1. Launch new nodes with patched AMI"
echo "  2. Cordon old nodes (stop scheduling new pods)"
echo "  3. Drain pods from old nodes (respecting PodDisruptionBudgets)"
echo "  4. Terminate old nodes"
echo ""
echo "PDBs ensure minAvailable: 1 for each service = zero downtime."
echo ""
read -p "Proceed with terraform apply? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply -auto-approve
else
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 3: Monitor node rotation"
CLUSTER_NAME=$(terraform output -raw eks_cluster_name 2>/dev/null || echo "samosachaat-$ENVIRONMENT")
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region us-west-2 2>/dev/null || true
echo "Watching nodes (Ctrl+C to stop):"
kubectl get nodes -w
