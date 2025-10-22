# Julia Transformer Ablation Experiments

This document describes how to run the Julia-based ablation sweeps that benchmark the Spiral Transformer adapter.

## Overview

The Julia implementation now provides a dedicated module, `SpiralTransformerAblation`, for generating randomised inputs, executing the transformer forward pass, and collecting gate/attention statistics.  A thin Python wrapper (`spiral_transformer_julia.load_ablation_module`) exposes the module so that experiments can be orchestrated either from Python or directly from Julia.

Experiment assets live under `experiments/julia_ablation/` and are designed to run outside of the exploratory notebooks.  The default configuration performs a small grid sweep over layer counts and feed-forward multipliers and stores the aggregated metrics as CSV files beneath `results/ablation/`.

## Prerequisites

1. Install Julia (1.9 or newer is recommended).
2. Activate the project environment that ships with the repository:

   ```bash
   julia --project=native/julia -e 'using Pkg; Pkg.instantiate()'
   ```

   This step downloads the packages referenced by the Julia transformer and ablation modules.

## Running a Sweep

Invoke the sweep runner with Julia, pointing it to the configuration you wish to use (defaults to `sweep_config.toml`).

```bash
cd /path/to/SpiralReality-AIT
julia --project=native/julia experiments/julia_ablation/run_sweep.jl
```

If the `julia` executable is not available, the repository also ships with a Python entrypoint that reproduces the sweep logic using the NumPy transformer implementation.  The Python wrapper automatically falls back to this mode when Julia is missing, or you can force it explicitly:

```bash
python experiments/julia_ablation/run_sweep.py --backend python
```

The Python CLI prefers the Julia command-line interface when it is available, but it can also call directly into the Julia module through `juliacall` if the bridge is installed without a system `julia` binary.  Pass `--dry-run` to exercise the configuration without writing a CSV file, and use `--limit N` to restrict the number of parameter combinations when sanity-checking large sweeps.

Key behaviours:

- The configuration file defines parameter defaults, grid values, and run-time metadata (see `experiments/julia_ablation/sweep_config.toml`).
- A Cartesian product of the configured parameter arrays is generated.  Each combination is repeated according to the `repeats` value, with deterministic seed offsets to make results reproducible.
- Results are written to `results/ablation/<prefix>_<timestamp>.csv`.  The CSV contains the input parameters plus summary statistics for attention energy, gate masks, and output activations.

To override the configuration path, pass it as the first argument:

```bash
julia --project=native/julia experiments/julia_ablation/run_sweep.jl my_sweep.toml
```

## Result Format

The emitted CSV files include the following columns:

- `d_model`, `n_layers`, `n_heads`, `ff_multiplier`, `seq_len`, `gate_scale`, `seed`, `repeat`
- `attention_mean`, `attention_std`
- `gate_mask_mean`, `gate_mask_std`
- `output_mean`, `output_std`

These metrics map directly to the named tuple returned by the Julia ablation module and provide a compact summary of each trial.

## Working from Python

Python code can use `spiral_transformer_julia.load_ablation_module()` to gain direct access to the Julia ablation helpers if tighter integration is required.  The loader ensures the Julia project is only initialised once per process and exposes the same API used by the CLI runner.
