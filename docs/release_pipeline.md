# SpiralReality AIT Release Pipeline

This document describes the end-to-end release workflow covering Python wheel
generation, container images, and Hugging Face Hub publishing.

## Versioning Strategy

* Use semantic versioning (`MAJOR.MINOR.PATCH`).
* Tag releases with `vX.Y.Z` and let `setuptools_scm` derive the version for
  local builds. Override with the `--version` flag in build scripts when
  preparing pre-release artifacts.

## Pre-release Checklist

1. Ensure the working tree is clean and tests pass: `pytest -q`.
2. Update `CHANGELOG.md` (if applicable) and bump metadata in
   [`pyproject.toml`](../pyproject.toml) when introducing backwards-incompatible
   changes.
3. Regenerate wheels via:

   ```bash
   ./packaging/build_wheel.sh --version X.Y.Z
   ./packaging/build_manylinux_wheels.sh --version X.Y.Z
   ```

4. Verify the built wheels with `twine check dist/*`.
5. Build the runtime container and smoke-test the API:

   ```bash
   export IMAGE_TAG=ghcr.io/spiralreality/spiralreality-ait:X.Y.Z
   docker build -f docker/runtime.Dockerfile -t "$IMAGE_TAG" .
   docker run --rm -p 8000:8000 "$IMAGE_TAG" python -m \
     spiralreality_AIT_onepass_aifcore_integrated.integrated.api
   ```

6. Export artifacts to the Hugging Face Hub:

   ```bash
   python tools/hf_upload.py spiralreality/spiralreality-ait \
     --path dist --metadata docs/hf_metadata.json
   ```

7. Create a signed git tag `git tag -s vX.Y.Z` and push `git push --tags`.

## Continuous Integration

Adopt the proposed GitHub Actions workflow in
[`packaging/github-actions-ci.yml`](../packaging/github-actions-ci.yml) to
automate wheel builds and PyPI publishing on tagged releases.

## Hugging Face Authentication & Metadata

1. Log in using the CLI once: `huggingface-cli login`.
2. Alternatively, pass a token via `--token` or set the `HF_TOKEN` environment
   variable before invoking `tools/hf_upload.py`.
3. Provide a metadata JSON file with fields such as `model_card`, `tags`, and
   `datasets` to populate the Hub README. Example template in
   [`docs/hf_metadata.json`](./hf_metadata.json).
4. Install the helper dependency: `python -m pip install huggingface_hub`.

## Release Artifacts

| Artifact | Source | Destination |
|----------|--------|-------------|
| Python wheels | `dist/` | PyPI, Hugging Face Hub |
| Docker image | `docker/runtime.Dockerfile` | Container registry |
| Model/Data bundle | `dist/` or curated folder | Hugging Face Hub |

## Post-release Checklist

* Verify PyPI publication and installability: `pip install spiralreality-ait`.
* Confirm the container image exists and is runnable from the registry.
* Validate the Hugging Face Hub repository renders the expected metadata.
* Create/Update announcement notes and internal dashboards as needed.
