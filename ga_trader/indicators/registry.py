from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from ga_trader.indicators.spec import IndicatorSpec

@dataclass
class IndicatorRegistry:
    root: Path
    specs: Dict[str, IndicatorSpec]

    @staticmethod
    def from_root(root: Path) -> "IndicatorRegistry":
        specs_dir = root / "specs"
        specs = {}
        if specs_dir.exists():
            for p in sorted(specs_dir.glob("*.yaml")):
                raw = yaml.safe_load(p.read_text(encoding="utf-8"))
                if not raw:
                    continue
                spec = IndicatorSpec(
                    indicator_id=str(raw["indicator_id"]),
                    name=str(raw.get("name", raw["indicator_id"])),
                    description=str(raw.get("description", "")),
                    author=str(raw.get("author", "")),
                    license=str(raw.get("license", "")),
                    source=str(raw.get("source", str(p))),
                    timeframe=str(raw.get("timeframe", "trigger")),
                    parameters=dict(raw.get("parameters", {}) or {}),
                    outputs=dict(raw.get("outputs", {}) or {}),
                    python_impl=str(raw.get("python_impl", "")),
                    pine_snippet=str(raw.get("pine_snippet", "")),
                )
                specs[spec.indicator_id] = spec
        return IndicatorRegistry(root=root, specs=specs)

    def list(self) -> List[IndicatorSpec]:
        return list(self.specs.values())

    def get(self, indicator_id: str) -> IndicatorSpec:
        if indicator_id not in self.specs:
            raise KeyError(f"Unknown indicator_id: {indicator_id}")
        return self.specs[indicator_id]

