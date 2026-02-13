#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${1:-bdd100k-od:latest}
CONTAINER_NAME=${2:-bdd100k-od-dev}
GPU_ARGS=${GPU_ARGS:---gpus all}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
else
  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
fi
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

if [[ -n "${GPU_ARGS}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found. GPU access may not be available." >&2
  fi
  if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "Warning: NVIDIA Container Toolkit runtime not detected in docker info." >&2
  fi
fi

DOCKER_ARGS=(
  --name "${CONTAINER_NAME}"
  --rm
  -it
)

if [[ -n "${GPU_ARGS}" ]]; then
  DOCKER_ARGS+=(${GPU_ARGS})
fi

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  DOCKER_ARGS+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

DOCKER_ARGS+=(
  -e "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
  -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
  -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
  -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  -p 8501:8501
  --shm-size=2g
  -v "$(pwd)":/workspace
  -w /workspace
  -e PYTHONPATH=/workspace/src
  "${IMAGE_NAME}"
)

docker run "${DOCKER_ARGS[@]}"
