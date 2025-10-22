from __future__ import annotations

"""Optional external encoder adapters (Julia / R).

The NumPy transformer adapter remains the default, but this module probes for
externally implemented variants that expose a compatible Python API.  When
available they allow the rest of the system to leverage vendor-optimised
attention kernels while preserving the same end-to-end entry points.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class ExternalEncoderHandle:
    impl: Any
    backend: str = "julia"
    device: str = "cpu"

    def forward(self, *args: Any, **kwargs: Any):  # pragma: no cover - passthrough
        return self.impl.forward(*args, **kwargs)

    def tune_from_boundary(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        if hasattr(self.impl, "tune_from_boundary"):
            self.impl.tune_from_boundary(*args, **kwargs)

    def export_state(self) -> dict:  # pragma: no cover - passthrough
        if hasattr(self.impl, "export_state"):
            return dict(self.impl.export_state())
        if hasattr(self.impl, "state_dict"):
            return dict(self.impl.state_dict())
        return {}

    def load_state(self, state: dict) -> None:  # pragma: no cover - passthrough
        if hasattr(self.impl, "load_state"):
            self.impl.load_state(state)
        elif hasattr(self.impl, "load_state_dict"):
            self.impl.load_state_dict(state)


def _candidate_modules() -> Tuple[Tuple[str, str], ...]:
    return (
        ("spiral_transformer_cpp", "CppTransformerAdapter"),
        ("spiral_transformer_julia", "JuliaTransformerAdapter"),
        ("spiral_transformer_r", "RTransformerAdapter"),
        ("spiralreality_transformer_cpp", "TransformerAdapter"),
        ("spiralreality_transformer_julia", "TransformerAdapter"),
        ("spiralreality_transformer_r", "TransformerAdapter"),
        ("spiral_transformer_adapter", "create_adapter"),
    )


def load_external_adapter(d_model: int, n_layers: int, seed: int) -> Optional[ExternalEncoderHandle]:
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
            impl = factory(d_model=d_model, n_layers=n_layers, seed=seed) if callable(factory) else factory
        except TypeError:
            try:
                impl = factory(d_model, n_layers, seed)  # pragma: no cover - legacy signature
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Failed to instantiate external encoder from %s", module_name)
                raise RuntimeError(f"Unable to instantiate external encoder: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to instantiate external encoder from %s", module_name)
            raise RuntimeError(f"Unable to instantiate external encoder: {exc}")
        backend = getattr(module, "BACKEND_KIND", getattr(impl, "backend", "julia"))
        device = getattr(module, "DEFAULT_DEVICE", getattr(impl, "device", "cpu"))
        LOGGER.info("Using %s transformer adapter from %s on %s", backend, module_name, device)
        return ExternalEncoderHandle(impl=impl, backend=str(backend), device=str(device))
    return None


def has_external_adapter() -> bool:
    for module_name, _ in _candidate_modules():
        try:
            importlib.import_module(module_name)
            return True
        except ModuleNotFoundError:
            continue
    return False
