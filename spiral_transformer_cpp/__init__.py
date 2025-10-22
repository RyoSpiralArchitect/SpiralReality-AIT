"""Wrapper module exposing the transformer adapter entry point."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import importlib
import importlib.util
import os

_spec = importlib.util.find_spec(f"{__name__}._spiral_transformer_cpp")
if _spec is None:  # pragma: no cover - pure Python fallback
    _native_mod = None
    _NativeAdapter = None
    BACKEND_KIND = "cpp"
    DEFAULT_DEVICE = "cpu"
    AVAILABLE_DEVICES = ("cpu",)
else:  # pragma: no cover - native module discovered
    _native_mod = importlib.import_module(f"{__name__}._spiral_transformer_cpp")
    _NativeAdapter = getattr(_native_mod, "CppTransformerAdapter", None)
    BACKEND_KIND = str(getattr(_native_mod, "BACKEND_KIND", "cpp-transformer"))
    DEFAULT_DEVICE = str(getattr(_native_mod, "DEFAULT_DEVICE", "cpu"))
    available = getattr(_native_mod, "AVAILABLE_DEVICES", (DEFAULT_DEVICE,))
    if isinstance(available, (list, tuple)):
        AVAILABLE_DEVICES = tuple(str(item) for item in available) or (DEFAULT_DEVICE,)
    else:
        AVAILABLE_DEVICES = (DEFAULT_DEVICE,)

from spiralreality_AIT_onepass_aifcore_integrated.integrated.encoder import (
    SpectralTransformerAdapter,
)


def _preferred_device(devices: Sequence[str]) -> Optional[str]:
    for candidate in devices:
        if str(candidate).lower() != "cpu":
            return str(candidate)
    if devices:
        return str(devices[0])
    return None


def _environment_device_request() -> Optional[str]:
    for key in ("SPIRAL_TRANSFORMER_DEVICE", "SPIRAL_DEVICE", "SPIRAL_DEFAULT_DEVICE"):
        value = os.getenv(key)
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _normalise_device_request(
    request: str, devices: Sequence[str], *, strict: bool
) -> Optional[str]:
    cleaned = request.strip()
    if not cleaned:
        return _preferred_device(devices)
    lowered = cleaned.lower()
    if lowered in {"auto", "default"}:
        return _preferred_device(devices)
    if lowered in {"gpu", "accelerator", "best"}:
        return _preferred_device(devices)
    token = lowered.split(":", 1)[0]
    for candidate in devices:
        if str(candidate).lower() == token:
            return str(candidate)
    if strict:
        available = ", ".join(str(dev) for dev in devices) or "cpu"
        raise ValueError(
            f"Unknown transformer device {request!r}. Available: {available}"
        )
    return None


class CppTransformerAdapter:
    """Delegate to the compiled adapter when available, otherwise reuse NumPy."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_multiplier: float = 4.0,
        seed: int = 2025,
        device: Optional[str] = None,
    ) -> None:
        preferred_device = None
        env_override = _environment_device_request()
        if device is not None:
            preferred_device = _normalise_device_request(
                device, AVAILABLE_DEVICES, strict=True
            )
        elif env_override is not None:
            preferred_device = _normalise_device_request(
                env_override, AVAILABLE_DEVICES, strict=False
            )
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
        if preferred_device is not None:
            setter = getattr(self._impl, "set_device", None) or getattr(self._impl, "to_device", None)
            if setter is not None:
                try:
                    setter(preferred_device)
                except Exception as exc:  # pragma: no cover - propagate configuration issues
                    raise ValueError(
                        f"Unable to set transformer device to {preferred_device!r}: {exc}"
                    ) from exc
            elif str(preferred_device).lower() not in {"cpu"}:
                raise ValueError("Device selection is not supported by the NumPy transformer")

    @property
    def backend(self) -> str:
        return getattr(self._impl, "backend", BACKEND_KIND)

    @property
    def device(self) -> str:
        return getattr(self._impl, "device", DEFAULT_DEVICE)

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
        return AVAILABLE_DEVICES


def create_adapter(
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ff_multiplier: float = 4.0,
    seed: int = 2025,
    device: Optional[str] = None,
) -> CppTransformerAdapter:
    return CppTransformerAdapter(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_multiplier=ff_multiplier,
        seed=seed,
        device=device,
    )
