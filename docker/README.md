# Docker Images

## Runtime Image

* **Base image:** `python:3.10-slim`
* **Entrypoint:** [`entrypoint.sh`](./entrypoint.sh) passes through the default
  command (`python -m spiralreality_AIT_onepass_aifcore_integrated.integrated.api`).
* **Dependencies:** Installs the wheel generated in `dist/` and runtime
  dependency `libgomp1`.

### Build and Push

```bash
export IMAGE_TAG=ghcr.io/spiralreality/spiralreality-ait:latest
docker build -f docker/runtime.Dockerfile -t "$IMAGE_TAG" .
docker push "$IMAGE_TAG"
```

## Manylinux Builder

The [`manylinux-build.Dockerfile`](./manylinux-build.Dockerfile) encapsulates
the build toolchain used by `packaging/build_manylinux_wheels.sh`.
