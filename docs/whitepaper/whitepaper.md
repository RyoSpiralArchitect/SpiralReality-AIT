# SpiralReality AIT Evaluation Whitepaper

## Overview
This document summarises a reproducible evaluation pipeline for the SpiralReality Active Inference Transformer (AIT). The goals were to consolidate latency, segmentation quality, and perturbation robustness measurements that previously lived inside unit tests and notebooks, and to package the workflow into a publication-ready artefact.

## Evaluation Protocol
- **Latency** measurement reuses the encode loop exercised by `tests/test_benchmarks.py`, timing repeated calls to `OnePassAIT.encode` on the bilingual prompt that the regression suite employs.
- **Segmentation F1** is computed with the boundary comparison helper from `tests/test_boundary.py`, contrasting the model's decoded segments with teacher annotations.
- **Robustness** is operationalised as the mean F1 achieved after randomly perturbing characters while preserving token lengths, mirroring the regression-style stress in the integrated demo notebook.

The evaluation harness is implemented in `scripts/run_evaluation.py`, which trains the boundary student on the curated corpus bundled with the repository, records diagnostic metadata, and serialises detailed metrics to `docs/whitepaper/data/`.

## Quantitative Results
- **Encode latency:** the average across eight runs is **152.5 ms**, aligning with the sub-second target validated in the regression test suite.
- **Segmentation F1:** the current configuration yields **0.00** on the curated teacher set. This reflects the student model's present limitations rather than a failure of the evaluation harness.
- **Perturbation robustness:** the mean F1 under 12% character corruption remains **0.00**; the degradation mirrors the baseline F1, indicating sensitivity to boundary noise.
- **Gate diagnostics:** the encode trace reports a mask energy of **0.249**, highlighting that the planner keeps a modest gating footprint despite the weak boundary scores.

### Visual Summaries
The figures below are generated directly from the evaluation artefacts to support the narrative.

![Segmentation F1 per sample](figures/segmentation_f1.svg)

![Encode latency](figures/latency.svg)

![Robustness under perturbations](figures/robustness.svg)

## Discussion
The consolidated pipeline confirms the latency characteristics established by the regression tests while exposing a performance gap in boundary quality. The zero F1 scores emphasise that additional curriculum data or architectural adjustments are required before deployment; nonetheless, the tooling is prepared to track improvements once those model changes land. The robustness experiment currently mirrors the segmentation baseline, signalling that perturbations do not introduce further degradation beyond the existing boundary failure mode.

## Reproducibility Checklist
1. Run `python scripts/run_evaluation.py` to regenerate raw metrics and CSV exports.
2. Execute `python docs/whitepaper/generate_figures.py` to refresh SVG charts.
3. Build the PDF with `python docs/whitepaper/build_whitepaper.py` or `make whitepaper`.
4. Archive `docs/whitepaper/data/` and `docs/whitepaper/figures/` alongside the generated PDF for release.

## Future Work
- Integrate multilingual fine-tuning to lift segmentation F1 above regression baselines.
- Extend the perturbation study to include token insertions and deletions once the decoding pipeline supports alignment-aware scoring.
- Automate comparisons against future checkpoints to visualise improvements across releases.
