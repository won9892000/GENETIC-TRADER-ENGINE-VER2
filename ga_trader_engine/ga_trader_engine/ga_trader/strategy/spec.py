from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Literal, Optional

FilterMode = Literal["absolute", "quantile"]

@dataclass(frozen=True)
class IndicatorFilter:
    indicator_id: str
    output: str
    comparator: str   # ">=" | "<="
    timeframe: str    # "trigger" | "anchor"

    # thresholding
    mode: FilterMode = "quantile"
    threshold: float = 0.0         # used when mode == "absolute"
    quantile: float = 0.5          # 0..1 used when mode == "quantile"
    lookback: int = 500            # bars used for percentile (Pine ta.percentile_linear_interpolation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "output": self.output,
            "comparator": self.comparator,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "threshold": float(self.threshold),
            "quantile": float(self.quantile),
            "lookback": int(self.lookback),
        }

@dataclass(frozen=True)
class StrategySpec:
    tf_anchor: str
    tf_trigger: str

    ema_fast: int
    ema_slow: int
    rsi_len: int
    rsi_long: float
    rsi_short: float
    atr_len: int
    sl_atr: float
    tp_atr: float
    z_len: int = 0
    z_entry: float = 0.0

    extra_filters: List[IndicatorFilter] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tf_anchor": self.tf_anchor,
            "tf_trigger": self.tf_trigger,
            "ema_fast": int(self.ema_fast),
            "ema_slow": int(self.ema_slow),
            "rsi_len": int(self.rsi_len),
            "rsi_long": float(self.rsi_long),
            "rsi_short": float(self.rsi_short),
            "atr_len": int(self.atr_len),
            "sl_atr": float(self.sl_atr),
            "tp_atr": float(self.tp_atr),
            "z_len": int(self.z_len),
            "z_entry": float(self.z_entry),
            "extra_filters": [f.to_dict() for f in (self.extra_filters or [])],
        }


def indicator_filter_from_dict(d: Dict[str, Any]) -> IndicatorFilter:
    return IndicatorFilter(
        indicator_id=str(d.get("indicator_id")),
        output=str(d.get("output")),
        comparator=str(d.get("comparator", ">=")),
        timeframe=str(d.get("timeframe", "trigger")),
        mode=str(d.get("mode", "quantile")),
        threshold=float(d.get("threshold", 0.0)),
        quantile=float(d.get("quantile", 0.5)),
        lookback=int(d.get("lookback", 500)),
    )

def strategy_spec_from_dict(d: Dict[str, Any]) -> StrategySpec:
    ef = d.get("extra_filters") or []
    filters = [indicator_filter_from_dict(x) for x in ef]
    dd = dict(d)
    dd["extra_filters"] = filters
    return StrategySpec(**dd)
