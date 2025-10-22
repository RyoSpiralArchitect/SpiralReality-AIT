# SpiralReality AIT Community Event Specification

## Overview
The SpiralReality AIT community event challenges participants to develop
streaming-friendly reasoning models that can be trained and evaluated in a
single pass over data. The competition builds directly on the public
benchmarks shipped with this repository so that results are reproducible and
aligned with the existing developer tooling.

The event focuses on three complementary tasks:

1. **Boundary Student Training** – participants improve segmentation-aware
   students trained through `OnePassAIT.train_student`. Reference behaviour is
   captured by `tests/test_boundary.py`.
2. **Streaming Encoder Efficiency** – entrants optimise
   `OnePassAIT.encode` latency, assessed using the regression test in
   `tests/test_benchmarks.py`.
3. **Multilingual Robustness** – solutions must retain performance when the
   teacher corpus is expanded using the multilingual utilities exercised in
   `tests/test_multilingual.py`.

## Data Sources
Participants must rely on the canonical corpora exposed in
`spiralreality_AIT_onepass_aifcore_integrated/integrated/corpus.py`. These
include:

- `TRAIN_TEXTS`: bilingual anchor passages used throughout the regression
  suite.
- `teacher_segments(texts)`: deterministic segmentations that act as ground
  truth for boundary quality metrics.
- Generated multilingual expansions from `build_multilingual_corpus`, allowing
  optional augmentation with reflective passages.

Additional ad-hoc data is allowed but final reports must disclose any external
sources and publish pre-processing pipelines to ensure reproducibility.

## Tasks and Milestones
| Phase | Deliverable | Details |
| --- | --- | --- |
| Warm-up | Baseline reproduction | Run the `tests/` suite unchanged and submit logs verifying parity with the reference implementation. |
| Sprint 1 | Boundary student enhancements | Demonstrate measurable segmentation improvements on the bilingual anchors while keeping latency within budget. |
| Sprint 2 | Streaming optimisation | Profile the encoder pipeline end-to-end and deliver patches that reduce average latency below the reference without regressing boundary accuracy. |
| Finale | Multilingual generalisation | Publish a model card reporting cross-lingual performance and resource usage across at least three language families. |

## Evaluation Metrics
Each submission will be measured against the following quantitative targets.

### Segmentation Quality
- **F1 score on bilingual anchors:** computed with the helper in
  `tests/test_boundary.py` using the deterministic `teacher_segments` labels.
- **Training stability:** monotonic decrease in `train_loss` across epochs, as
  asserted in the existing tests.

### Streaming Efficiency
- **Average encoding latency:** mean time per `OnePassAIT.encode` call across
  three runs must stay below 1.0 s for the warm-up prompt, matching the
  `EncodeLatencyTest` guard.
- **Gate diagnostics availability:** participants must expose phase and gate
  masks so downstream dashboards can surface inspection artefacts.

### Multilingual Robustness
- **Language coverage histogram:** submissions report the distribution of
  dataset tags returned by `language_histogram` to ensure balanced coverage.
- **Cross-lingual boundary fidelity:** compare segmentation F1 between
  original and multilingual corpora; a relative drop greater than 10% triggers
  a review.

### Reporting Requirements
- Publish a model card summarising dataset composition, training budgets,
  encoder backend selection, and hardware utilisation.
- Provide reproducible scripts (preferably via notebooks) that can be executed
  in headless CI environments.

## Submission Format
Teams must submit:

1. Source patches or configuration overrides targeting the modules in
   `spiralreality_AIT_onepass_aifcore_integrated/integrated/`.
2. A structured report (`reports/<team>.md`) including metric tables and
   qualitative analysis.
3. A CI artifact bundle containing raw logs, latency traces, and gate
   diagnostics exported as JSON.

## Leaderboard and Scoring
Scores will be computed as a weighted aggregate:

- 40% segmentation quality (primary F1 metric).
- 30% latency improvements relative to the reference median.
- 20% multilingual robustness (harmonic mean of per-language F1).
- 10% documentation completeness and reproducibility evidence.

Leaderboard updates will be published weekly. Ties are broken by the earliest
submission that passes the full CI pipeline without manual intervention.

## Compliance Checklist
Before submitting, ensure the following:

- [ ] All unit tests in `tests/` pass locally.
- [ ] Latency traces demonstrate <1.0 s encode time on the provided prompt.
- [ ] Model card and run logs are added to the submission bundle.
- [ ] Any external data usage is declared with licensing information.
- [ ] Safety review completed for multilingual content.

