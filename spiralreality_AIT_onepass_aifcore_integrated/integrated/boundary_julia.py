from __future__ import annotations

"""Optional bridge to Julia/R-backed boundary student implementations.

This module mirrors :mod:`boundary_cpp` but targets runtimes that expose a
Python interop surface via PyJulia or rpy2.  When a compatible backend is not
available the loader simply returns ``None`` so the NumPy student remains the
source of truth.
"""

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .np_compat import np

LOGGER = logging.getLogger(__name__)


def _normalise_devices(candidate: Any) -> Tuple[str, ...]:
    if candidate is None:
        return ("cpu",)
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:  # pragma: no cover - defensive
            return ("cpu",)
    if isinstance(candidate, (list, tuple, set)):
        items = list(candidate)
    elif isinstance(candidate, str):
        items = [candidate]
    else:
        try:
            items = list(candidate)
        except Exception:  # pragma: no cover - defensive
            return ("cpu",)
    result: List[str] = []
    for item in items:
        if item is None:
            continue
        result.append(str(item))
    return tuple(result or ("cpu",))


@dataclass
class JuliaStudentHandle:
    """Wrap a dynamically loaded Julia/R boundary student implementation."""

    impl: Any
    backend: str = "julia"
    device: str = "cpu"
    devices: Tuple[str, ...] = field(default_factory=lambda: ("cpu",))

    def configure(self, cfg_dict: Dict[str, Any]) -> None:
        if hasattr(self.impl, "configure"):
            self.impl.configure(cfg_dict)

    def attach_phase(self, phase: Any) -> None:
        if hasattr(self.impl, "attach_phase"):
            self.impl.attach_phase(phase)

    def attach_encoder(self, encoder: Any) -> None:
        if hasattr(self.impl, "attach_encoder"):
            self.impl.attach_encoder(encoder)

    def train(self, texts: Sequence[str], segments: Sequence[Sequence[str]], cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.impl, "train"):
            raise RuntimeError("Julia boundary student does not expose a train method")
        summary = self.impl.train(texts, segments, cfg_dict)
        if isinstance(summary, dict):
            summary = dict(summary)
            summary.setdefault("backend", f"{self.backend}:{self.device}")
        return summary

    def boundary_probs(self, text: str) -> np.ndarray:
        if not hasattr(self.impl, "boundary_probs"):
            raise RuntimeError("Julia boundary student missing boundary_probs")
        probs = self.impl.boundary_probs(text)
        if hasattr(probs, "detach"):
            probs = probs.detach()
        if hasattr(probs, "cpu"):
            probs = probs.cpu()
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        return np.array(list(probs), dtype=float)

    def decode(self, text: str) -> Sequence[str]:
        if not hasattr(self.impl, "decode"):
            raise RuntimeError("Julia boundary student missing decode")
        output = self.impl.decode(text)
        if isinstance(output, (str, bytes)):
            return [str(output)]
        return list(output)

    def export_state(self) -> Dict[str, Any]:
        if hasattr(self.impl, "state_dict"):
            return dict(self.impl.state_dict())
        if hasattr(self.impl, "export_state"):
            return dict(self.impl.export_state())
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        if hasattr(self.impl, "load_state_dict"):
            self.impl.load_state_dict(state)
        elif hasattr(self.impl, "load_state"):
            self.impl.load_state(state)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def available_devices(self) -> Tuple[str, ...]:
        if hasattr(self.impl, "available_devices"):
            try:
                return _normalise_devices(self.impl.available_devices())
            except Exception:  # pragma: no cover - defensive
                return self.devices or (self.device,)
        if hasattr(self.impl, "devices"):
            try:
                return _normalise_devices(getattr(self.impl, "devices"))
            except Exception:  # pragma: no cover - defensive
                return self.devices or (self.device,)
        return self.devices or (self.device,)

    def preferred_device(self) -> str:
        devices = self.available_devices()
        for dev in devices:
            lowered = dev.lower()
            if any(lowered.startswith(prefix) for prefix in ("cuda", "gpu", "metal")):
                return dev
        return devices[0] if devices else self.device

    def to_device(self, device: str) -> bool:
        setter = getattr(self.impl, "to_device", None) or getattr(self.impl, "set_device", None)
        try:
            if callable(setter):
                setter(device)
                self.device = device
                return True
        except Exception:  # pragma: no cover - defensive
            return False
        try:
            setattr(self.impl, "device", device)
            self.device = device
            return True
        except Exception:  # pragma: no cover - defensive
            return False


_JULIA_CACHE: Optional[JuliaStudentHandle] = None


def _candidate_modules() -> Iterable[Tuple[str, str]]:
    """Yield module / attribute name pairs to probe."""

    yield ("spiral_boundary_julia", "JuliaBoundaryStudent")
    yield ("spiral_boundary_r", "RBoundaryStudent")
    yield ("spiralreality_boundary_julia", "BoundaryStudent")
    yield ("spiralreality_boundary_r", "BoundaryStudent")
    yield ("spiral_boundary_student", "create_student")


def _ensure_runtime_ready() -> None:
    """Attempt to eagerly import PyJulia/rpy2 when present.

    Import errors are swallowed—the loader will simply keep probing the
    candidate backends.  This makes it safe to run in pure-Python environments
    while still giving compiled runtimes a chance to register themselves.
    """

    try:  # pragma: no cover - optional dependency hint
        importlib.import_module("julia")
    except ModuleNotFoundError:
        pass
    try:  # pragma: no cover - optional dependency hint
        importlib.import_module("rpy2")
    except ModuleNotFoundError:
        pass


def load_julia_student() -> Optional[JuliaStudentHandle]:
    """Return a handle to an external Julia/R backend if available."""

    global _JULIA_CACHE
    if _JULIA_CACHE is not None:
        return _JULIA_CACHE

    _ensure_runtime_ready()

    for module_name, attr in _candidate_modules():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        factory = getattr(module, attr, None)
        if factory is None:
            LOGGER.warning("Found module %s but missing %s attribute", module_name, attr)
            continue
        try:
            impl = factory() if callable(factory) else factory
        except Exception as exc:  # pragma: no cover - defensive log path
            LOGGER.exception("Failed to instantiate Julia boundary student from %s", module_name)
            raise RuntimeError(f"Unable to instantiate Julia boundary student: {exc}")
        backend = getattr(module, "BACKEND_KIND", getattr(impl, "backend", "julia"))
        device = getattr(module, "DEFAULT_DEVICE", getattr(impl, "device", "cpu"))
        devices = ("cpu",)
        for attr in ("AVAILABLE_DEVICES", "DEVICES", "SUPPORTED_DEVICES"):
            if hasattr(module, attr):
                devices = _normalise_devices(getattr(module, attr))
                break
        else:
            if hasattr(impl, "available_devices"):
                try:
                    devices = _normalise_devices(impl.available_devices())
                except Exception:  # pragma: no cover - defensive
                    devices = (str(device),)
            elif hasattr(impl, "devices"):
                devices = _normalise_devices(getattr(impl, "devices"))
            else:
                devices = _normalise_devices(device)
        LOGGER.info(
            "Using %s boundary student backend from %s on %s (devices=%s)",
            backend,
            module_name,
            device,
            ",".join(devices),
        )
        _JULIA_CACHE = JuliaStudentHandle(impl=impl, backend=str(backend), device=str(device), devices=devices)
        return _JULIA_CACHE
    return None


def has_julia_backend() -> bool:
    return load_julia_student() is not None


def julia_backend_devices() -> Tuple[str, ...]:
    handle = load_julia_student()
    if handle is None:
        return ()
    return handle.available_devices()
