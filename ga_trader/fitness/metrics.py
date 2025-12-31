from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

def normalize_01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))

@dataclass(frozen=True)
class FitnessSpec:
    mode: str
    weights: Dict[str, float]
    constraints: Dict[str, float]

