#!/usr/bin/env python3
"""Run Julia ablation sweeps with a Python fallback implementation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import itertools
import pathlib
import shutil
import sys
from typing import Any, Iterable, Iterator, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spiralreality_AIT_onepass_aifcore_integrated.integrated import np_compat

np = np_compat.np

try:  # Optional dependency – only present when the Julia bridge is installed.
    from spiral_transformer_julia import load_ablation_module
except ModuleNotFoundError:  # pragma: no cover - runtime guard when Julia is absent
    load_ablation_module = None  # type: ignore[assignment]

from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import (
    SpectralTransformerAdapter,
)


CONFIG_PATH_DEFAULT = pathlib.Path(__file__).with_name("sweep_config.toml")
RESULTS_DIR_DEFAULT = pathlib.Path(__file__).parent.parent.parent / "results" / "ablation"


@dataclasses.dataclass
class AblationSpec:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_multiplier: float = 4.0
    seq_len: int = 32
    gate_scale: float = 1.0
    seed: int = 2025
    repeat: int = 1


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_defaults(cfg: dict[str, Any]) -> AblationSpec:
    return AblationSpec(
        d_model=_coerce_int(cfg.get("d_model"), 128),
        n_layers=_coerce_int(cfg.get("n_layers"), 4),
        n_heads=_coerce_int(cfg.get("n_heads"), 4),
        ff_multiplier=_coerce_float(cfg.get("ff_multiplier"), 4.0),
        seq_len=_coerce_int(cfg.get("seq_len"), 32),
        gate_scale=_coerce_float(cfg.get("gate_scale"), 1.0),
        seed=_coerce_int(cfg.get("seed"), 2025),
        repeat=_coerce_int(cfg.get("repeat"), 1),
    )


def _parameter_grid_items(parameters: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if not parameters:
        yield {}
        return
    keys = list(parameters.keys())
    values = [parameters[key] for key in keys]
    expanded: list[Sequence[Any]] = []
    for value in values:
        if isinstance(value, str) or not isinstance(value, Iterable):
            expanded.append([value])
        else:
            expanded.append(list(value))
    for combo in itertools.product(*expanded):
        yield {key: combo[idx] for idx, key in enumerate(keys)}


def expand_overrides(grid: Iterable[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    for index, base_override in enumerate(grid):
        for repeat in range(1, repeats + 1):
            entry = dict(base_override)
            entry["repeat"] = repeat
            entry["seed_offset"] = index * repeats + (repeat - 1)
            overrides.append(entry)
    return overrides


def _materialise_spec(base: AblationSpec, overrides: dict[str, Any]) -> AblationSpec:
    spec = AblationSpec(
        d_model=_coerce_int(overrides.get("d_model", base.d_model), base.d_model),
        n_layers=_coerce_int(overrides.get("n_layers", base.n_layers), base.n_layers),
        n_heads=_coerce_int(overrides.get("n_heads", base.n_heads), base.n_heads),
        ff_multiplier=_coerce_float(
            overrides.get("ff_multiplier", base.ff_multiplier), base.ff_multiplier
        ),
        seq_len=_coerce_int(overrides.get("seq_len", base.seq_len), base.seq_len),
        gate_scale=_coerce_float(
            overrides.get("gate_scale", base.gate_scale), base.gate_scale
        ),
        seed=_coerce_int(overrides.get("seed", base.seed), base.seed),
        repeat=_coerce_int(overrides.get("repeat", base.repeat), base.repeat),
    )
    seed_offset = _coerce_int(overrides.get("seed_offset"), 0)
    spec.seed += seed_offset
    return spec


def _random_inputs(spec: AblationSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    X = rng.normal(0.0, 1.0, (spec.seq_len, spec.d_model))
    if spec.seq_len <= 1:
        gate_pos = np.zeros((spec.seq_len,), dtype=float)
    else:
        denom = max(spec.seq_len - 1, 1)
        step = spec.gate_scale / denom
        gate_values = [step * idx for idx in range(spec.seq_len)]
        gate_pos = np.array(gate_values, dtype=float)
    return X, gate_pos


def _flatten_values(data: Any) -> list[float]:
    if hasattr(data, "to_list"):
        return _flatten_values(data.to_list())
    if isinstance(data, (list, tuple)):
        out: list[float] = []
        for item in data:
            out.extend(_flatten_values(item))
        return out
    try:
        return [float(data)]
    except (TypeError, ValueError):
        return []


def _attn_stats(attn_layers: Sequence[Any]) -> tuple[float, float]:
    if not attn_layers:
        return 0.0, 0.0
    layer_means: list[float] = []
    for layer in attn_layers:
        values = _flatten_values(layer)
        if not values:
            continue
        abs_vals = [abs(v) for v in values]
        layer_means.append(float(sum(abs_vals) / len(abs_vals)))
    if not layer_means:
        return 0.0, 0.0
    data = np.array(layer_means, dtype=float)
    return float(np.mean(data)), float(np.std(data))


def _mask_stats(mask: Any) -> tuple[float, float]:
    if mask is None:
        return 0.0, 0.0
    values = _flatten_values(mask)
    if not values:
        return 0.0, 0.0
    data = np.array(values, dtype=float)
    return float(np.mean(data)), float(np.std(data))


def _output_stats(output: Any) -> tuple[float, float]:
    values = _flatten_values(output)
    if not values:
        return 0.0, 0.0
    abs_vals = np.array([abs(v) for v in values], dtype=float)
    return float(np.mean(abs_vals)), float(np.std(abs_vals))


def _run_single_python(spec: AblationSpec) -> dict[str, Any]:
    adapter = SpectralTransformerAdapter(
        d_model=spec.d_model,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        ff_multiplier=spec.ff_multiplier,
        seed=spec.seed,
    )
    X, gate_pos = _random_inputs(spec)
    output = adapter.forward(X, gate_pos)
    attn_mean, attn_std = _attn_stats(adapter.last_attn)
    mask_mean, mask_std = _mask_stats(adapter.last_gate_mask)
    out_mean, out_std = _output_stats(output)
    return {
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "ff_multiplier": spec.ff_multiplier,
        "seq_len": spec.seq_len,
        "gate_scale": spec.gate_scale,
        "seed": spec.seed,
        "repeat": spec.repeat,
        "attention_mean": attn_mean,
        "attention_std": attn_std,
        "gate_mask_mean": mask_mean,
        "gate_mask_std": mask_std,
        "output_mean": out_mean,
        "output_std": out_std,
    }


def _run_julia_module(overrides: list[dict[str, Any]], base: AblationSpec) -> list[dict[str, Any]]:
    if load_ablation_module is None:
        raise RuntimeError("Julia ablation module is not available")
    module = load_ablation_module()
    jl_base = module.AblationSpec(
        d_model=base.d_model,
        n_layers=base.n_layers,
        n_heads=base.n_heads,
        ff_multiplier=base.ff_multiplier,
        seq_len=base.seq_len,
        gate_scale=base.gate_scale,
        seed=base.seed,
        repeat=base.repeat,
    )
    rows = module.run_parameter_sweep(overrides, base=jl_base)
    results: list[dict[str, Any]] = []
    for row in rows:
        current: dict[str, Any] = {}
        for key in row.keys():
            value = row[key]
            if isinstance(value, (int, float)):
                current[str(key)] = float(value)
            else:
                current[str(key)] = value
        results.append(current)
    return results


def _write_results(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(header, "") for header in headers])


def _ensure_results_dir(raw_dir: str | None) -> pathlib.Path:
    base = pathlib.Path(raw_dir) if raw_dir else RESULTS_DIR_DEFAULT
    if not base.is_absolute():
        base = (pathlib.Path(__file__).parent / base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_python_backend(
    overrides: list[dict[str, Any]],
    base: AblationSpec,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for overrides_entry in overrides:
        spec = _materialise_spec(base, overrides_entry)
        results.append(_run_single_python(spec))
    return results


def call_julia_cli(julia_path: pathlib.Path, project: pathlib.Path, config_path: pathlib.Path) -> int:
    command = [str(julia_path), f"--project={project}", str(pathlib.Path(__file__).with_name("run_sweep.jl")), str(config_path)]
    completed = __import__("subprocess").run(command, check=False)
    return completed.returncode


def parse_config(path: pathlib.Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=CONFIG_PATH_DEFAULT,
        help="Path to the sweep configuration file (default: sweep_config.toml)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "julia", "python"],
        default="auto",
        help="Execution backend. 'auto' prefers Julia CLI when available",
    )
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=pathlib.Path("native/julia"),
        help="Julia project directory passed to --project when using the Julia CLI",
    )
    parser.add_argument(
        "--julia-bin",
        type=pathlib.Path,
        default=None,
        help="Explicit path to the julia executable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the sweep but skip writing the CSV output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N parameter combinations",
    )

    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = parse_config(config_path)
    defaults_cfg = dict(config.get("defaults", {}))
    parameters_cfg = dict(config.get("parameters", {}))
    experiment_cfg = dict(config.get("experiment", {}))

    base = load_defaults(defaults_cfg)
    grid = list(_parameter_grid_items(parameters_cfg))
    repeats = _coerce_int(experiment_cfg.get("repeats", 1), 1)
    overrides = expand_overrides(grid, repeats)
    if args.limit is not None:
        if args.limit <= 0:
            overrides = []
        else:
            overrides = overrides[: args.limit]

    backend_choice = args.backend
    julia_path_raw = args.julia_bin or shutil.which("julia")
    backend_mode: str
    if backend_choice == "auto":
        if julia_path_raw:
            backend_mode = "julia-cli"
        elif load_ablation_module is not None:
            backend_mode = "julia-module"
        else:
            backend_mode = "python"
    elif backend_choice == "julia":
        if julia_path_raw:
            backend_mode = "julia-cli"
        elif load_ablation_module is not None:
            backend_mode = "julia-module"
        else:
            print("Julia runtime not available; falling back to Python backend.")
            backend_mode = "python"
    else:
        backend_mode = "python"

    if backend_mode == "julia-cli" and julia_path_raw:
        if args.limit is not None:
            print("Warning: --limit is not supported by the Julia CLI backend.")
        return call_julia_cli(pathlib.Path(julia_path_raw), args.project, config_path)

    if backend_mode == "julia-module":
        results = _run_julia_module(overrides, base)
    else:
        results = run_python_backend(overrides, base)

    if args.dry_run:
        print(f"Computed {len(results)} result rows (dry run; no file written).")
        return 0

    results_dir_raw = experiment_cfg.get("results_dir")
    results_dir = _ensure_results_dir(results_dir_raw)
    prefix = experiment_cfg.get("file_prefix", "ablation")
    output_path = results_dir / f"{prefix}_{_timestamp()}.csv"
    _write_results(output_path, results)
    print(f"Wrote {len(results)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
