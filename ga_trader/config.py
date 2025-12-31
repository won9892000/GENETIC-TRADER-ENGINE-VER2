from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pathlib import Path

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]
    path: Path

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path)
        return Config(raw=load_yaml(p), path=p)

    # Convenience accessors (minimal; callers can use cfg.raw too)
    @property
    def timeframes(self) -> List[str]:
        return list(get(self.raw, "data.timeframes", ["5m"]))

    @property
    def run_root(self) -> Path:
        return Path(get(self.raw, "project.run_root", "runs"))

    @property
    def run_name(self) -> str:
        return str(get(self.raw, "project.run_name", "latest"))

    @property
    def parity_symbol(self) -> str | None:
        # Symbol used for Python↔TradingView parity trade export/validation (optional).
        val = get(self.raw, "validation.parity_symbol", None)
        if val in (None, "", []):
            val = get(self.raw, "project.parity_symbol", None)
        if val in (None, ""):
            return None
        return str(val)

    @property
    def commission_bps(self) -> float:
        return float(get(self.raw, "fees.commission_bps", 0.0))

