#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <docker-hub-repo> [tag] [--no-push]"
    echo ""
    echo "Examples:"
    echo "  $0 myuser/nanochat-runpod"
    echo "  $0 myuser/nanochat-runpod v1.0"
    echo "  $0 myuser/nanochat-runpod latest --no-push"
    exit 1
}

[[ $# -lt 1 ]] && usage

DOCKER_REPO="$1"
TAG="${2:-latest}"
NO_PUSH=false

for arg in "$@"; do
    [[ "$arg" == "--no-push" ]] && NO_PUSH=true
done

IMAGE="${DOCKER_REPO}:${TAG}"

echo ">>> Building ${IMAGE} ..."

docker build -t "$IMAGE" "$SCRIPT_DIR"

echo ">>> Built ${IMAGE}"

if [[ "$NO_PUSH" == true ]]; then
    echo ">>> Skipping push (--no-push)"
else
    echo ">>> Pushing ${IMAGE} ..."
    docker push "$IMAGE"
    echo ">>> Pushed ${IMAGE}"
fi
