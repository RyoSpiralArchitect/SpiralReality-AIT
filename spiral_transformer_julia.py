"""Julia-backed transformer adapter loader.

This optional module allows :mod:`integrated.encoder_backends` to discover the
Julia transformer implementation when ``juliacall`` is available.  When the
Julia runtime is missing the import raises :class:`ModuleNotFoundError` so the
NumPy transformer remains the default.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from juliacall import Main as jl
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ModuleNotFoundError(
        "juliacall is required to use the Julia transformer adapter"
    ) from exc


_MODULE_CACHE: dict[str, object] = {}


def _module_path(module_name: str) -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parent
    julia_dir = repo_root / "native" / "julia"
    return julia_dir / f"{module_name}.jl"


def load_julia_module(module_name: str = "SpiralTransformerJulia") -> object:
    """Return a cached Julia module, loading it on first use."""

    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    module_path = _module_path(module_name)
    if not module_path.exists():  # pragma: no cover - defensive runtime branch
        raise FileNotFoundError(f"Missing Julia module: {module_path}")

    jl.include(str(module_path))
    try:
        module = getattr(jl, module_name)
    except AttributeError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            f"Julia module '{module_name}' did not register in Main"
        ) from exc

    _MODULE_CACHE[module_name] = module
    return module


def _module() -> object:
    return load_julia_module("SpiralTransformerJulia")


BACKEND_KIND = "julia-transformer"
DEFAULT_DEVICE = "cpu"


class JuliaTransformerAdapter:
    """Thin Python wrapper over the Julia transformer implementation."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_multiplier: float = 4.0,
        seed: int = 2025,
    ) -> None:
        module = _module()
        self._impl = module.create_adapter(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_multiplier=ff_multiplier,
            seed=seed,
        )
        self.backend = getattr(module, "BACKEND_KIND", BACKEND_KIND)
        self.device = getattr(module, "DEFAULT_DEVICE", DEFAULT_DEVICE)
        self.last_attn: list[np.ndarray] = []
        self.last_gate_mask: np.ndarray = np.zeros((0, 0), dtype=float)

    def forward(
        self,
        X: np.ndarray,
        gate_pos: np.ndarray,
        gate_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        module = _module()
        gate_mask_arg = gate_mask if gate_mask is not None else jl.nothing
        result = np.array(module.forward(self._impl, X, gate_pos, gate_mask=gate_mask_arg), dtype=float)
        attn_list = jl.getproperty(self._impl, "last_attn")
        self.last_attn = [np.array(arr, dtype=float) for arr in attn_list]
        mask = jl.getproperty(self._impl, "last_gate_mask")
        self.last_gate_mask = np.array(mask, dtype=float)
        self.device = getattr(self._impl, "device", self.device)
        return result

    def tune_from_boundary(
        self, base_gate: Iterable[float], targets: Iterable[float], lr: float = 1e-3
    ) -> None:
        module = _module()
        tune_fn = getattr(module, "tune_from_boundary!", None)
        if tune_fn is not None:
            tune_fn(self._impl, list(base_gate), list(targets), lr=lr)

    def export_state(self) -> dict:
        module = _module()
        state = module.export_state(self._impl)

        def _coerce(value):
            try:
                return np.array(value, dtype=float).tolist()
            except TypeError:
                if isinstance(value, (list, tuple)):
                    return [_coerce(v) for v in value]
                return value

        return {str(k): _coerce(v) for k, v in dict(state).items()}

    def load_state(self, state: dict) -> None:
        module = _module()
        load_fn = getattr(module, "load_state!", None)
        if load_fn is not None:
            load_fn(self._impl, state)

    def device_inventory(self) -> Sequence[str]:
        module = _module()
        devices = module.device_inventory(self._impl)
        return [str(dev) for dev in devices]


try:  # pragma: no cover - optional discovery
    AVAILABLE_DEVICES = tuple(JuliaTransformerAdapter().device_inventory())
except Exception:  # pragma: no cover - defensive fallback
    AVAILABLE_DEVICES = (DEFAULT_DEVICE,)


def load_ablation_module() -> object:
    """Return the Julia ablation helper module."""

    return load_julia_module("SpiralTransformerAblation")


def create_adapter(
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ff_multiplier: float = 4.0,
    seed: int = 2025,
) -> JuliaTransformerAdapter:
    return JuliaTransformerAdapter(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_multiplier=ff_multiplier,
        seed=seed,
    )


__all__ = [
    "BACKEND_KIND",
    "DEFAULT_DEVICE",
    "AVAILABLE_DEVICES",
    "JuliaTransformerAdapter",
    "create_adapter",
    "load_julia_module",
    "load_ablation_module",
]
