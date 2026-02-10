#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${1:-bdd100k-od:latest}

docker build -t "${IMAGE_NAME}" .
