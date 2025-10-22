from __future__ import annotations

# Optional bridge to compiled boundary student backends.
# When a compiled backend is unavailable we fall back to the NumPy trainer.
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .np_compat import np

LOGGER = logging.getLogger(__name__)


def _normalise_devices(candidate: Any) -> Tuple[str, ...]:
    """Return a tuple of device identifiers from a backend hint."""

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
    normalised: List[str] = []
    for item in items:
        if item is None:
            continue
        normalised.append(str(item))
    return tuple(normalised or ("cpu",))


@dataclass
class CompiledStudentHandle:
    """Wrap a dynamically loaded compiled student implementation."""

    impl: Any
    device: str = "cpu"
    backend: str = "cpp"
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
            raise RuntimeError("Compiled student does not expose a train method")
        return self.impl.train(texts, segments, cfg_dict)

    def boundary_probs(self, text: str) -> np.ndarray:
        if not hasattr(self.impl, "boundary_probs"):
            raise RuntimeError("Compiled student missing boundary_probs")
        probs = self.impl.boundary_probs(text)
        if hasattr(probs, "detach"):
            probs = probs.detach().cpu().numpy()
        return np.array(list(probs), dtype=float)

    def decode(self, text: str) -> Sequence[str]:
        if not hasattr(self.impl, "decode"):
            raise RuntimeError("Compiled student missing decode")
        return list(self.impl.decode(text))

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


_COMPILED_CACHE: Optional[CompiledStudentHandle] = None


def _candidate_modules() -> Iterable[str]:
    yield "spiral_boundary_gpu"
    yield "spiralreality_boundary_gpu"
    yield "spiral_boundary_cpp"
    yield "spiralreality_boundary_cpp"
    yield "spiralreality_cpp.boundary"
    yield "boundary_student_ext"


def load_compiled_student() -> Optional[CompiledStudentHandle]:

    global _COMPILED_CACHE
    if _COMPILED_CACHE is not None:
        return _COMPILED_CACHE
    for name in _candidate_modules():
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        StudentClass = getattr(module, "CppBoundaryStudent", None)
        if StudentClass is None:
            LOGGER.warning("Found module %s but no CppBoundaryStudent class", name)
            continue
        try:
            impl = StudentClass()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to instantiate compiled boundary student from %s", name)
            raise RuntimeError(f"Unable to instantiate compiled boundary student: {exc}")
        device = getattr(module, "DEFAULT_DEVICE", getattr(impl, "device", "cpu"))
        backend = getattr(module, "BACKEND_KIND", getattr(impl, "backend", "cpp"))
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
            "Using compiled boundary student backend from %s on %s (devices=%s)",
            name,
            device,
            ",".join(devices),
        )
        _COMPILED_CACHE = CompiledStudentHandle(impl=impl, device=str(device), backend=str(backend), devices=devices)
        return _COMPILED_CACHE
    return None


def has_compiled_backend() -> bool:
    return load_compiled_student() is not None


def compiled_backend_devices() -> Tuple[str, ...]:
    handle = load_compiled_student()
    if handle is None:
        return ()
    return handle.available_devices()
