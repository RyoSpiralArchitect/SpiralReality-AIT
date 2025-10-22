
This build **replaces AIF's world-model step with the One‑Pass AIT dynamics** and now trains the
boundary student end-to-end with a tiny NN+CRF head tied into the encoder.

## Highlights
- Hybrid boundary learner: learnable char embeddings → shallow tanh block → binary CRF with
  Viterbi decoding.  The head trains jointly with the SpectralTransformerAdapter via a
  lightweight feedback rule and keeps learnable phase bases for gating.  Optional bridges in
  `integrated/boundary_cpp.py` and `integrated/boundary_julia.py` now expose device discovery so
  bespoke C++/Julia/R implementations can advertise CUDA/Metal targets and accept `device_preference`
  hints while falling back to the pure NumPy student when unavailable.
- Spectral transformer encoder: `integrated/encoder.py` upgrades the old toy adapter to a
  multi-head, layer-normalised NumPy transformer with phase-aware FiLM modulation.  It keeps track
  of gate-aware attention maps and reports its device inventory so external backends (or future GPU
  shims) can slot in without touching the One-Pass logic.
- Streaming-friendly trainer: `StudentTrainingConfig(cache_sequences=False)` turns on
  low-memory mode so the boundary learner rebuilds sequences on the fly, enabling massive
  corpora to be processed in pure NumPy without staging every example in RAM. Training summaries
  report cache usage alongside backend/device inventories so large-scale jobs remain observable.
- Phase-aware encoding: curvature-derived local features are folded into the positional signal and
  boundary probabilities seed a gated attention mask, lifting stability and F1 on reflective text.
  The encoder loader (`integrated/encoder_backends.py`) also probes for Julia/R transformer
  adapters, mirroring the boundary student's backend selection so joint training can ride compiled
  kernels.
- Learned latent dynamics: a small MLP (see `integrated/dynamics.py`) distils the handcrafted
  transition rule and powers `OnePassAIT.predict_next` once sufficient experience has been
  collected.
- Deployment ready: a FastAPI server (`integrated/api.py`) exposes `/segment`, `/encode`, `/train`
  and `/load` endpoints, and `integrated/checkpoint.py` serialises model state to JSON.
- Instrumentation: `OnePassAIT.gate_diagnostics()` surfaces gate traces, attention energy, and gate
  mask strength.  `integrated/run_demo.py` streams TensorBoard scalars (JSON fallback) for training
  loss/F1, latency, gate energy, and phase statistics while persisting checkpoints/logs for
  inspection.
- Visualization ready: `notebooks/run_demo.ipynb` mirrors the demo with Matplotlib/Seaborn hooks for
  attention maps, phase traces, and gate overlays.
- Curated corpora: the demo now assembles a multilingual dataset that augments the reflective
  English/Japanese anchors with curated Spanish, French, German, and Chinese narratives. The helper
  utilities in `integrated/multilingual.py` register the segments so the trainer and tests can reuse
  them consistently while exposing language histograms and per-language length/token statistics for
  rapid dataset audits.

## Layout
- `integrated/aif_core/` — compact Active Inference Core v2.
- `integrated/onepass_ait.py` — learnable phase basis, boundary NN+CRF, latent dynamics, diagnostics.
- `integrated/boundary.py` / `phase.py` / `encoder.py` / `dynamics.py` — modular components powering
  the student and latent model.
- `integrated/gwm_bridge.py` — binds One‑Pass AIT to AIF (ctx & step hooks).
- `integrated/run_demo.py` — end‑to‑end run; writes `integrated_log.json`, TensorBoard scalars, and
  a checkpoint.
- `notebooks/run_demo.ipynb` — interactive variant of the demo with visualization scaffolding.
- `tests/` — segmentation quality + encode latency regression tests.
- `.github/workflows/ci.yml` — GitHub Actions workflow (compile check + unit tests).
- `native/cpp` — pybind11 implementation of the boundary student (`spiral_boundary_cpp`).
- `native/julia` — Julia boundary student (`SpiralBoundaryJulia`) for PyJulia/rpy2 bridges.

## Quick run
```bash
python -m integrated.run_demo
```

The demo trains the boundary student on the multilingual corpus by default. Use
`OnePassAIT.train_student(languages=("es", "ja"), include_reflective=False)` to target specific
languages programmatically.

Artifacts:
- `integrated_log.json` → chosen actions, EFE aggregates, belief updates, segmentation metrics,
  gate diagnostics.
- `logs/` → TensorBoard event files or JSON scalars describing training/evaluation traces.
- `checkpoint.json` → JSON checkpoint for reloading through the FastAPI service.

## REST API (optional)
```bash
uvicorn spiralreality_AIT_onepass_aifcore_integrated.integrated.api:create_app --factory
```

Endpoints: `/health`, `/train`, `/segment`, `/encode`, `/load`.

## Tests & CI
```bash
python -m unittest discover -v
```

CI runs the unit tests plus a `compileall` lint on Python 3.11.

## Native backends

Optional compiled backends live under `native/`.  The C++ variant builds a
`spiral_boundary_cpp` extension with pybind11; see `native/cpp/README.md` for
instructions.  The Julia implementation in `native/julia/SpiralBoundaryJulia.jl`
exposes a `JuliaBoundaryStudent` that matches the Python contract and can be
loaded through PyJulia.  When either module is on the Python path the loader in
`integrated/boundary_{cpp,julia}.py` will activate it automatically and the
NumPy trainer becomes a safety net rather than the primary implementation.
