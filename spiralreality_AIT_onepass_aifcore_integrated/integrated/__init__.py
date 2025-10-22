"""SpiralReality AIT integrated package."""

from importlib.metadata import PackageNotFoundError, version


try:  # pragma: no cover - metadata is provided at build time
    __version__ = version("spiralreality-ait")
except PackageNotFoundError:  # pragma: no cover - fallback during development
    __version__ = "0.0.0.dev0"


__all__ = [
    "__version__",
]
