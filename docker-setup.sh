#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${1:-bdd100k-od:latest}
CONTAINER_NAME=${2:-bdd100k-od-dev}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

docker run --name "${CONTAINER_NAME}" --rm -it \
  -p 8501:8501 \
  --shm-size=2g \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  "${IMAGE_NAME}"
