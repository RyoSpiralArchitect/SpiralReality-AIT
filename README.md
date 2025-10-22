# integrated (onepass AIT + aif_core) — Overview and Quick Start

This build **replaces AIF's world-model step with the One‑Pass AIT dynamics** and now trains the
boundary student end-to-end with a tiny NN+CRF head tied into the encoder.

## Highlights
- Hybrid boundary learner: learnable char embeddings → shallow tanh block → binary CRF with
  Viterbi decoding.  The head trains jointly with the SpectralTransformerAdapter via a
  lightweight feedback rule and keeps learnable phase bases for gating.  Optional bridges in
  `integrated/boundary_cpp.py` and `integrated/boundary_julia.py` now expose device discovery so
  bespoke C++/Julia/R implementations can advertise CUDA/Metal targets and accept `device_preference`
  hints while falling back to the pure NumPy student when unavailable.  The compiled C++ stub
  honours `SPIRAL_BOUNDARY_DEVICE` (falling back to `SPIRAL_DEVICE` or `SPIRAL_DEFAULT_DEVICE`) so
  deployments can steer default accelerator selections without touching the training code.
- Spectral transformer encoder: `integrated/encoder.py` upgrades the old toy adapter to a
  multi-head, layer-normalised NumPy transformer with phase-aware FiLM modulation.  It keeps track
  of gate-aware attention maps and reports its device inventory so external backends (or future GPU
  shims) can slot in without touching the One-Pass logic.  The optional
  `spiral_transformer_cpp` build now installs alongside the Python wrapper and surfaces
  compile-time CUDA/ROCm/MPS hints through `device_inventory()`/`set_device()` so GPU-capable
  targets can be selected without adding a PyTorch dependency while honouring
  `SPIRAL_TRANSFORMER_DEVICE`/`SPIRAL_DEVICE` overrides when choosing a default accelerator.
- Streaming-friendly trainer: `StudentTrainingConfig(cache_sequences=False)` turns on
  low-memory mode so the boundary learner rebuilds sequences on the fly, enabling massive
  corpora to be processed in pure NumPy without staging every example in RAM. Training summaries
  report cache usage alongside backend/device inventories so large-scale jobs remain observable.
- Phase-aware encoding: curvature-derived local features are folded into the positional signal and
  boundary probabilities seed a gated attention mask, lifting stability and F1 on reflective text.
  The encoder loader (`integrated/encoder_backends.py`) also probes for C++/Julia/R transformer
  adapters, mirroring the boundary student's backend selection so joint training can ride compiled
  kernels.
- Learned latent dynamics: a small MLP (see `integrated/dynamics.py`) distils the handcrafted
  transition rule and powers `OnePassAIT.predict_next` once sufficient experience has been
  collected.
- Deployment ready: a lightweight, dependency-free server (`server/main.py`) exposes `/health`
  and a WebSocket stream for boundary diagnostics, while `integrated/checkpoint.py` serialises
  model state to JSON for scripted usage.
- Instrumentation: `OnePassAIT.gate_diagnostics()` surfaces gate traces, attention energy, and gate
  mask strength.  `integrated/run_demo.py` streams structured JSON scalars for training loss/F1,
  latency, gate energy, and phase statistics while persisting checkpoints/logs for inspection.
- Visualization ready: `notebooks/run_demo.ipynb` mirrors the demo with Matplotlib/Seaborn hooks for
  attention maps, phase traces, and gate overlays.
- Curated corpora: the demo now assembles a multilingual dataset that augments the reflective
  English/Japanese anchors with curated Spanish, French, German, and Chinese narratives. The helper
  utilities in `integrated/multilingual.py` register the segments so the trainer and tests can reuse
  them consistently while exposing language histograms and per-language length/token statistics for
  rapid dataset audits.
- Licensed dataset export: `integrated/corpus.py` exposes `corpus_license()`/`corpus_catalog()` so the
  reflective and multilingual corpora can be redistributed under CC‑BY‑4.0 with per-language
  summaries for reporting or downstream tooling.
- Robustness benchmarking: `integrated/benchmark.py` trains the boundary student, applies dialect,
  noise, and tempo perturbations via `integrated/augmentation.py`, reports segmentation F1, p95
  latency, np_stub vs NumPy error, and writes JSON/Markdown summaries for dashboards.

## Layout
- `integrated/aif_core/` — compact Active Inference Core v2.
- `integrated/onepass_ait.py` — learnable phase basis, boundary NN+CRF, latent dynamics, diagnostics.
- `integrated/boundary.py` / `phase.py` / `encoder.py` / `dynamics.py` — modular components powering
  the student and latent model.
- `integrated/gwm_bridge.py` — binds One‑Pass AIT to AIF (ctx & step hooks).
- `integrated/run_demo.py` — end‑to‑end run; writes `integrated_log.json`, scalar logs, and a
  checkpoint.
- `notebooks/run_demo.ipynb` — interactive variant of the demo with visualization scaffolding.
- `tests/` — segmentation quality + encode latency regression tests.
- `.github/workflows/ci.yml` — GitHub Actions workflow (compile check + unit tests).
- `integrated/run_demo.py` — end‑to‑end run; writes `integrated_log.json` and a checkpoint.
- `tests/` — segmentation quality + encode latency regression tests.
- `.github/workflows/ci.yml` — GitHub Actions workflow (compile check + unit tests).
  
## Overview

This directory contains an integration of the "onepass" text-processing experiments with an aif_core component. The implementation is primarily NumPy-based and demonstrates a one-pass (online) processing pipeline that combines segmentation (boundary detection), phase-based local features, and a toy transformer-style encoder to produce contextualized embeddings.
tention

## Main files

- onepass_ait.py
  - Core implementation. Contains BoundaryStudent (boundary detector), phase feature computation, ToyTransformerAdapter (a minimal attention-style encoder), and OnePassAIT (integration logic).
- run_demo.py
  - Demo / example runner that shows how to run onepass_ait on sample text and visualize or print outputs.
- gwm_bridge.py
  - A bridging wrapper for connecting to external modules or alternative implementations.
- aif_core/
  - A directory for related core functionality if present; see that directory for more details.

## Minimal dependencies

- Python 3.9+
- numpy

Install example:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy
```

The demo trains the boundary student on the multilingual corpus by default. Use
`OnePassAIT.train_student(languages=("es", "ja"), include_reflective=False)` to target specific
languages programmatically.

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `logs/` → JSONL scalar logs describing training/evaluation traces.
## Quick start

1. Create and activate a virtual environment and install dependencies (see above).
2. Run the demo from the repository root:

```bash
python spiralreality_AIT_onepass_aifcore_integrated/integrated/run_demo.py
```

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `checkpoint.json` → JSON checkpoint for reloading through the diagnostics service.

## Packaging and Distribution

* Python packaging is configured via [`pyproject.toml`](./pyproject.toml). Use
  the helper scripts in [`packaging/`](./packaging) to build wheels locally or
  inside the manylinux Docker image.
* Build a wheel on the host:

  ```bash
  ./packaging/build_wheel.sh --version 0.1.0
  ```

* Produce manylinux wheels:

  ```bash
  ./packaging/build_manylinux_wheels.sh --version 0.1.0
  ```

## Docker Images

The [`docker/`](./docker) directory contains the runtime and manylinux builder
Dockerfiles. Build and push the runtime image with:

```bash
export IMAGE_TAG=ghcr.io/spiralreality/spiralreality-ait:latest
docker build -f docker/runtime.Dockerfile -t "$IMAGE_TAG" .
docker push "$IMAGE_TAG"
```

Refer to [`docker/README.md`](./docker/README.md) for more details.

Endpoints: `/health`, `/train`, `/segment`, `/encode`, `/load`.

## Tests & CI
```bash
python -m unittest discover -v
```

The demo trains the boundary student on the multilingual corpus by default. Use
`OnePassAIT.train_student(languages=("es", "ja"), include_reflective=False)` to target specific
languages programmatically.

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `logs/` → JSONL scalar logs describing training/evaluation traces.
- `checkpoint.json` → JSON checkpoint for reloading through the diagnostics service.

## Whitepaper Evaluation Pipeline

The repository ships with a reproducible workflow for building the latency/F1/robustness report in
`docs/whitepaper/`.

1. Generate raw metrics and CSV exports:

   ```bash
   python scripts/run_evaluation.py
   ```

2. Produce SVG figures (pure-Python implementation, no external plotting stack required):

   ```bash
   python docs/whitepaper/generate_figures.py
   ```

3. Build the PDF whitepaper:

   ```bash
   # Requires matplotlib >= 3.7. Install via `pip install matplotlib`.
   python docs/whitepaper/build_whitepaper.py
   ```

   Alternatively, the whole pipeline can be executed with a single command:

   ```bash
   make whitepaper
   ```

The scripts emit artefacts into `docs/whitepaper/data/` and `docs/whitepaper/figures/`. A release
checklist describing publication gating, DOI management, and GitHub Release hygiene is available at
`docs/whitepaper/release_checklist.md`.


## Native backends

Optional compiled backends live under `native/`.  The C++ variant builds a
`spiral_boundary_cpp` extension with pybind11; see `native/cpp/README.md` for
instructions.  A companion `spiral_boundary_gpu` module mirrors the API while
surfacing compiled accelerator targets (CUDA, ROCm/HIP, Metal) to Python so the
runtime can route training telemetry through native code even before GPU
kernels land.  The Julia implementation in `native/julia/SpiralBoundaryJulia.jl`
performs a similar device probe for `CUDA.jl`, `AMDGPU.jl`, and `Metal.jl` and
exposes the selected device in its summaries.  When any compiled module is on
the Python path the loader in `integrated/boundary_{cpp,julia}.py` will activate
it automatically and the NumPy trainer becomes a safety net rather than the
primary implementation.

## Real-time diagnostics stack

A lightweight, pure-Python diagnostics server and Vite dashboard are included
for streaming boundary inference:

```bash
docker compose up --build
```

- Backend service: `server/main.py` exposes `/health` and a `/ws` WebSocket that
  streams boundary segments plus `GateDiagnostics` metrics for arbitrary text
  without requiring external Python packages.
- Frontend app: `frontend/` connects to the WebSocket, renders gate traces and
  boundary probabilities, and can be served locally via `npm run dev`.
- Compose demo: `docker-compose.yml` wires both images for a one-command local
  experience. The dashboard becomes available at <http://localhost:5173> with
  the API at <http://localhost:8000>.

To run the diagnostics server without Docker execute:

```bash
python -m server.main
```

For a notebook walkthrough see [`notebooks/TUTORIAL.md`](notebooks/TUTORIAL.md)
and [`notebooks/websocket_demo.ipynb`](notebooks/websocket_demo.ipynb).
