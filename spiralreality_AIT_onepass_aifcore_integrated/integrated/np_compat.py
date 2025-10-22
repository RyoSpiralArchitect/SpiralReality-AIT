"""Compatibility layer that now always exposes real NumPy.

Earlier iterations of the project shipped a tiny pure Python replacement when
NumPy was not available.  The refined implementation assumes the scientific
stack is present, so importing this module will raise immediately if NumPy
cannot be resolved.
"""

from __future__ import annotations

import numpy as np  # type: ignore

HAS_NUMPY = True

__all__ = ["np", "HAS_NUMPY"]
