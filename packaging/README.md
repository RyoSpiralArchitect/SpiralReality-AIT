# Packaging and Build Overview

This repository did not previously include a `setup.cfg` or `pyproject.toml`.
The new [`pyproject.toml`](../pyproject.toml) configures a setuptools based
build for the `spiralreality_AIT_onepass_aifcore_integrated.integrated`
package. Version metadata is resolved via `setuptools_scm`, falling back to
`0.0.1` for ad-hoc builds.

## Build scripts

* [`build_wheel.sh`](./build_wheel.sh) builds an sdist and wheel into the
  `dist/` directory using isolated environments.
* [`build_manylinux_wheels.sh`](./build_manylinux_wheels.sh) drives the
  `docker/manylinux-build.Dockerfile` image to produce manylinux-compatible
  artifacts.

Install the `build` package locally (`python -m pip install build`) before
invoking the scripts outside of CI.

Both scripts accept an optional `--version` argument to set a specific build
version via the `SETUPTOOLS_SCM_PRETEND_VERSION` environment variable.

## Continuous integration

The [`github-actions-ci.yml`](./github-actions-ci.yml) workflow proposal runs
linting, unit tests, and publishes wheels to GitHub Releases when a tag is
created. Adapt the release steps to suit your secrets management strategy.
