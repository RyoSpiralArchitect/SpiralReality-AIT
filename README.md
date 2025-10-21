
This build **replaces AIF's world-model step with the One‑Pass AIT dynamics** and now trains the
boundary student end-to-end with a tiny NN+CRF head tied into the encoder.

## Highlights
- Hybrid boundary learner: learnable char embeddings → shallow tanh block → binary CRF with
  Viterbi decoding.  The head trains jointly with the ToyTransformerAdapter via a lightweight
  feedback rule and keeps learnable phase bases for gating.  A compiled backend loader hooks into
  torch-style C++/Julia/R modules when present, falling back to the pure NumPy trainer otherwise.
- Phase-aware encoding: curvature-derived local features are folded into the positional signal and
  boundary probabilities seed a gated attention mask, lifting stability and F1 on reflective text.
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
- Curated corpus: the demo seeds the student with over twenty reflective English prompts (plus the
  bilingual originals) so the CRF head learns boundary cues rooted in investigative workflows.

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

## Quick run
```bash
python -m integrated.run_demo
```

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
