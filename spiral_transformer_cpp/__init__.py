"""Wrapper module exposing the transformer adapter entry point."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

try:  # pragma: no cover - optional native module path
    from ._spiral_transformer_cpp import CppTransformerAdapter as _NativeAdapter
except ModuleNotFoundError:  # pragma: no cover - pure Python fallback
    _NativeAdapter = None

from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import (
    SpectralTransformerAdapter,
)


class CppTransformerAdapter:
    """Delegate to the compiled adapter when available, otherwise reuse NumPy."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_multiplier: float = 4.0,
        seed: int = 2025,
    ) -> None:
        if _NativeAdapter is not None:
            self._impl = _NativeAdapter(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ff_multiplier=ff_multiplier,
                seed=seed,
            )
        else:
            self._impl = SpectralTransformerAdapter(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ff_multiplier=ff_multiplier,
                seed=seed,
            )

    @property
    def backend(self) -> str:
        return getattr(self._impl, "backend", "cpp")

    @property
    def device(self) -> str:
        return getattr(self._impl, "device", "cpu")

    @property
    def last_attn(self) -> Sequence:
        return getattr(self._impl, "last_attn", [])

    @property
    def last_gate_mask(self):  # type: ignore[override]
        return getattr(self._impl, "last_gate_mask", None)

    def forward(self, X, gate_pos, gate_mask: Optional = None):
        if gate_mask is None:
            return self._impl.forward(X, gate_pos)
        return self._impl.forward(X, gate_pos, gate_mask=gate_mask)

    def tune_from_boundary(
        self, base_gate: Iterable[float], targets: Iterable[float], lr: float = 1e-3
    ) -> None:
        if hasattr(self._impl, "tune_from_boundary"):
            self._impl.tune_from_boundary(base_gate, targets, lr)

    def export_state(self) -> dict:
        if hasattr(self._impl, "export_state"):
            return dict(self._impl.export_state())
        return {}

    def load_state(self, state: dict) -> None:
        if hasattr(self._impl, "load_state"):
            self._impl.load_state(state)

    def device_inventory(self) -> Sequence[str]:
        if hasattr(self._impl, "device_inventory"):
            return tuple(self._impl.device_inventory())
        return (self.device,)


BACKEND_KIND = "cpp"
DEFAULT_DEVICE = "cpu"
AVAILABLE_DEVICES = ("cpu",)


def create_adapter(
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ff_multiplier: float = 4.0,
    seed: int = 2025,
) -> CppTransformerAdapter:
    return CppTransformerAdapter(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_multiplier=ff_multiplier,
        seed=seed,
    )
