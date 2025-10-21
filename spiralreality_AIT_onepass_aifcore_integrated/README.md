# One‑Pass AIT × AIF Core v2 — Fully Integrated (GWM bridged)

This build **replaces AIF's world-model step with the One‑Pass AIT dynamics**:
- AIT (Student‑internalized Adapter/FiLM) encodes the prompt once → `H`, `r2_local`, boundary probs.
- For planning, AIF's **step function** calls AIT's `predict_next(μ,Σ, action, ctx)`.
- The **context vector** comes from policy‑specific local gates over `H` (ctx_provider).
- `R3_mix` is updated by AIT as the plan rolls out (approx schedule for epistemic weight).

## What you get
- `integrated/aif_core/` — compact AIF Core v2
- `integrated/onepass_ait.py` — Student head + tiny Transformer + AIT one‑pass adapter
- `integrated/gwm_bridge.py` — bridge that binds AIT to AIF (ctx & step)
- `integrated/run_demo.py` — end‑to‑end run; outputs `integrated_log.json`

## Quick run
```bash
python -m integrated.run_demo
```

Artifacts: `integrated_log.json` contains chosen actions, EFE aggregates, belief updates, and R3_mix.
