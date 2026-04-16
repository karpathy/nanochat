#!/usr/bin/env bash
set -euo pipefail

# Verify all samosaChaat services are healthy after deployment.
# Usage: ./scripts/verify-deployment.sh <namespace>

NAMESPACE="${1:?Usage: verify-deployment.sh <namespace>}"

echo "=== samosaChaat Deployment Verification — $NAMESPACE ==="

PASS=0
FAIL=0

check() {
    local name="$1" cmd="$2"
    if eval "$cmd" > /dev/null 2>&1; then
        echo "  ✓ $name"
        ((PASS++))
    else
        echo "  ✗ $name"
        ((FAIL++))
    fi
}

echo ""
echo "Pods:"
kubectl get pods -n "$NAMESPACE" --no-headers | while read line; do
    echo "  $line"
done

echo "Health checks:"
check "Frontend" "kubectl exec -n $NAMESPACE deploy/frontend -- wget -qO- http://localhost:3000/api/health"
check "Auth" "kubectl exec -n $NAMESPACE deploy/auth -- python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8001/auth/health')\""
check "Chat API" "kubectl exec -n $NAMESPACE deploy/chat-api -- python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8002/api/health')\""
check "Inference" "kubectl exec -n $NAMESPACE deploy/inference -- python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8003/health')\""

echo ""
echo "Deployments:"
check "Frontend available" "kubectl rollout status deploy/frontend -n $NAMESPACE --timeout=10s"
check "Auth available" "kubectl rollout status deploy/auth -n $NAMESPACE --timeout=10s"
check "Chat API available" "kubectl rollout status deploy/chat-api -n $NAMESPACE --timeout=10s"
check "Inference available" "kubectl rollout status deploy/inference -n $NAMESPACE --timeout=10s"

echo ""
echo "PDBs:"
kubectl get pdb -n "$NAMESPACE" --no-headers 2>/dev/null | while read line; do
    echo "  $line"
done

echo ""
echo "Result: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "All checks passed!" || { echo "SOME CHECKS FAILED"; exit 1; }
