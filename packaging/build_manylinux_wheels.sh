#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

VERSION=""
IMAGE_TAG="spiralreality-ait-manylinux:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BUILD_ARGS=()
if [[ -n "${VERSION}" ]]; then
  BUILD_ARGS+=("--build-arg" "PACKAGE_VERSION=${VERSION}")
fi

docker build \
  -f "${REPO_ROOT}/docker/manylinux-build.Dockerfile" \
  "${REPO_ROOT}" \
  -t "${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}"

RUN_ENV=("-v" "${REPO_ROOT}:/workspace")
if [[ -n "${VERSION}" ]]; then
  RUN_ENV+=("-e" "SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}")
fi

docker run --rm \
  "${RUN_ENV[@]}" \
  "${IMAGE_TAG}" \
  /bin/bash -c "cd /workspace && /opt/python/cp39-cp39/bin/pip wheel . -w dist/"
