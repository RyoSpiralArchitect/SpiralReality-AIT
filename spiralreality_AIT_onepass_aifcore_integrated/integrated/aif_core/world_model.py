from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ActionSpace:
    names: list[str]
    vecs: dict[str, np.ndarray]

    def all_actions(self) -> list[str]:
        return list(self.names)

    def vec(self, a: str) -> np.ndarray:
        return self.vecs[a]
