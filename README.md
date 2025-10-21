# integrated (onepass AIT + aif_core) — Overview and Quick Start

## Overview

This directory contains an integration of the "onepass" text-processing experiments with an aif_core component. The implementation is primarily NumPy-based and demonstrates a one-pass (online) processing pipeline that combines segmentation (boundary detection), phase-based local features, and a toy transformer-style encoder to produce contextualized embeddings.

Goals for now:
- Proof of concept for an online (single-pass) text processing pipeline
- Investigate using boundary detection and phase information to gate attention
- Provide a lightweight, easy-to-visualize experimental implementation

## Purpose of this directory

- Collect the integrated prototype that combines onepass AIT logic and aif_core-related functionality
- Provide demo scripts and a bridge for connecting to other modules or environments
- Serve as a starting point for migrating to a learnable model and more robust pipeline

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

(If migrating to PyTorch later, add `torch` to dependencies.)

Install example:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy
```

## Quick start

1. Create and activate a virtual environment and install dependencies (see above).
2. Run the demo from the repository root:

```bash
python spiralreality_AIT_onepass_aifcore_integrated/integrated/run_demo.py
```

3. Check console output and any generated plots or files. The demo prints boundary probabilities, phase curvature, gate positions, and encoder outputs.

### About run_demo

- run_demo.py instantiates onepass_ait.OnePassAIT and runs sample text through the encode pipeline. It typically prints or visualizes boundary probabilities (`ps`), local phase curvature (`r2_local`), gate positions (`gate_pos`), and the encoder output (`H`).

## Development and extension suggestions (priority order)

Short-term (easy wins):
- Convert BoundaryStudent to a small PyTorch module (or CRF) and enable end-to-end training.
- Turn run_demo into a Jupyter Notebook to visualize attention, phase traces, and gates.
- Add unit tests and CI (GitHub Actions) to protect refactors.

Medium-term:
- Replace the toy encoder with a PyTorch Transformer encoder and incorporate `gate_pos` as an attention bias or gating mechanism.
- Replace seeded/fixed embeddings with learnable embeddings (nn.Embedding) or subword tokenization.
- Replace the simple `predict_next` dynamics with a learned transition model (MLP/RNN/KalmanNet-style).

Long-term (production-oriented):
- Add model checkpointing (save/load), an inference API (FastAPI), and GPU support.
- Build a batch training and evaluation pipeline for larger datasets.

## Evaluation and visualization

Short-term evaluation:
- Segmentation metrics: Precision / Recall / F1 using labeled segment boundaries.
- Visualize attention maps, phase traces for each character, and `gate_pos` over time to inspect behavior.

Tools: matplotlib, seaborn, tensorboard (optional).

## Contributing

- Small contributions (docs, tests, demos) are welcome.
- For larger changes (PyTorch migration, API work), please open an issue to discuss the design before submitting a major PR.

## Contact / Notes

- Repository owner: @RyoSpiralArchitect
- Key files and experiments to check: run_demo.py, gwm_bridge.py
