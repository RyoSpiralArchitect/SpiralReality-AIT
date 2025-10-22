# Online Learning Pipeline Requirements

This document summarises the functional and technical requirements for the
proposed online learning pipeline and identifies extension points within the
existing implementation under
`spiralreality_AIT_onepass_aifcore_integrated/integrated/`.

## Goals
- Support **streaming ingestion** of prompts and token-level annotations without
  requiring full dataset materialisation.
- Maintain **single-pass training guarantees** for the boundary student and
  latent phase models.
- Enable **incremental evaluation** so that leaderboard metrics can be updated
  in near real-time.
- Provide hooks for **adaptive scheduling** (e.g., prioritised replay or
  curriculum adjustments) while remaining backwards compatible with existing
  APIs.

## Reference Architecture
The current modules already expose the core stages required for an online
pipeline:

1. **Ingestion** – `BoundaryStudent.train` internally consumes sequences
   provided through `OnePassAIT.train_student`. The helper `teacher_segments`
   pre-computes boundary references which can be replaced by a streaming
   iterator.
2. **Encoding** – `SpectralTransformerAdapter.forward` processes character-level
   embeddings and gate masks incrementally, maintaining cached attention maps in
   `self.last_attn` and `self.last_gate_mask` for downstream diagnostics.
3. **Latent Dynamics** – `LatentDynamicsModel` and `PhaseBasisLearner` evolve the
   latent state and curvature statistics, ensuring continuity across batches.
4. **Evaluation** – `OnePassAIT.encode` returns both gating signals and phase
   projections that can feed latency and segmentation quality monitors.
5. **Serving** – `api.create_app` exposes minimal FastAPI endpoints for train,
   encode, and segment operations, acting as a control plane for incoming
   events.

## Identified Extension Points
| Component | File / Symbol | Extension Strategy |
| --- | --- | --- |
| Ingestion controller | `onepass_ait.py::train_student` | Accept asynchronous generators for `texts`/`segments` and schedule commits to `BoundaryStudent` once mini-batch statistics have been updated. |
| Segment cache | `boundary.py::BoundaryStudent.train` | Replace in-memory `cache_sequences` storage with a configurable streaming buffer (e.g., LMDB or Redis streams) to handle long-running sessions. |
| Encoder instrumentation | `encoder.py::SpectralTransformerAdapter.forward` | Emit structured telemetry (attention heatmaps, gate energy) via callbacks so the online dashboard can react without blocking the forward pass. |
| Phase updates | `phase.py::PhaseBasisLearner.update` | Introduce partial updates keyed by sequence ID, enabling interleaved multilingual inputs while keeping the EWMA decay (`beta`) stable. |
| API integration | `api.py::create_app` | Add WebSocket endpoints for incremental evaluations and integrate authentication/queueing middleware for competition submissions. |

## Required Enhancements
1. **Streaming corpus adapters** – Implement iterables that wrap `teacher_segments`
   outputs into chunked streams with backpressure control. This allows loading
   slices from `TRAIN_TEXTS` or multilingual expansions without full copies.
2. **Checkpoint rotation** – Extend `checkpoint.py` to support time-based
   snapshots triggered after a configurable number of streamed updates.
3. **Latency budget enforcement** – Use the timing harness in
   `tests/test_benchmarks.py` to validate that asynchronous pipelines stay below
   the 1.0 s encode target.
4. **Event queue integration** – Define a contract for external message brokers
   (Kafka, Redis Streams) to push new samples; add translation layers that feed
   the existing `train_student` call.
5. **Live metric aggregation** – Store per-update metrics (segmentation F1,
   gate mask energy) in a rolling window so the leaderboard can display trend
   lines instead of just batch-level scores.

## Non-Functional Requirements
- **Observability:** Export Prometheus-compatible counters for encode latency,
  gate mask energy, and phase curvature changes.
- **Fault tolerance:** Ensure partial updates can be replayed by idempotent
  checkpoint loads from `load_checkpoint`.
- **Privacy:** Sensitive multilingual corpora must be filtered to respect
  licensing and user-consent constraints; integration with anonymisation passes
  is required before persistence.
- **Scalability:** Encoder backends enumerated via `encoder_backends.py` should
  be hot-swappable so specialised hardware can be enabled for finalists.

