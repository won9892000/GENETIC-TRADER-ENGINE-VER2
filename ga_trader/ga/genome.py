from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import random

from ga_trader.utils import clip
from ga_trader.strategy.spec import StrategySpec, IndicatorFilter
from ga_trader.indicators.registry import IndicatorRegistry

@dataclass
class Genome:
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
    z_len: int
    z_entry: float

    extra_filters: List[IndicatorFilter]

    def to_spec(self) -> StrategySpec:
        return StrategySpec(
            tf_anchor=self.tf_anchor,
            tf_trigger=self.tf_trigger,
            ema_fast=int(self.ema_fast),
            ema_slow=int(self.ema_slow),
            rsi_len=int(self.rsi_len),
            rsi_long=float(self.rsi_long),
            rsi_short=float(self.rsi_short),
            atr_len=int(self.atr_len),
            sl_atr=float(self.sl_atr),
            tp_atr=float(self.tp_atr),
            z_len=int(self.z_len),
            z_entry=float(self.z_entry),
            extra_filters=list(self.extra_filters or []),
        )

def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    return int(rng.randint(int(lo), int(hi)))

def _rand_float(rng: random.Random, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())

def _random_filter(rng: random.Random, reg: IndicatorRegistry, space: Dict[str, Any] | None = None) -> IndicatorFilter:
    specs = reg.list()
    if not specs:
        raise ValueError("No indicators available in registry")
    ind = rng.choice(specs)
    out = rng.choice(list(ind.outputs.keys()))
    comparator = rng.choice([">=", "<="])
    timeframe = str(ind.timeframe or "trigger")

    # Default: quantile-based thresholding for cross-asset scale robustness.
    q_lo, q_hi = 0.1, 0.9
    lb_lo, lb_hi = 200, 1500
    if space:
        if "filter_quantile" in space:
            q_lo, q_hi = float(space["filter_quantile"][0]), float(space["filter_quantile"][1])
        if "filter_lookback" in space:
            lb_lo, lb_hi = int(space["filter_lookback"][0]), int(space["filter_lookback"][1])
    quantile = float(q_lo + (q_hi - q_lo) * rng.random())
    lookback = int(rng.randint(int(lb_lo), int(lb_hi)))

    # Choose filter mode: quantile (default) or absolute (optional mix).
    abs_prob = 0.0
    if space and "filter_absolute_prob" in space:
        try:
            abs_prob = float(space["filter_absolute_prob"])
        except Exception:
            abs_prob = 0.0
    abs_prob = max(0.0, min(1.0, abs_prob))

    # If output has a scale range, absolute thresholds can be sampled meaningfully.
    out_cfg = ind.outputs.get(out, {}) if isinstance(ind.outputs, dict) else {}
    scale = out_cfg.get("scale")
    can_absolute = isinstance(scale, (list, tuple)) and len(scale) == 2 and scale[0] is not None and scale[1] is not None

    if rng.random() < abs_prob and can_absolute:
        lo, hi = float(scale[0]), float(scale[1])
        if lo > hi:
            lo, hi = hi, lo
        thr = float(lo + (hi - lo) * rng.random())
        return IndicatorFilter(
            indicator_id=ind.indicator_id,
            output=out,
            comparator=comparator,
            timeframe=timeframe,
            mode="absolute",
            threshold=thr,
            quantile=quantile,
            lookback=lookback,
        )

    # quantile mode (cross-asset robust default)
    return IndicatorFilter(
        indicator_id=ind.indicator_id,
        output=out,
        comparator=comparator,
        timeframe=timeframe,
        mode="quantile",
        quantile=quantile,
        lookback=lookback,
        threshold=0.0,
    )

def random_genome(space: Dict[str, Any], rng: random.Random, timeframes: List[str], reg: IndicatorRegistry | None) -> Genome:
    tf_anchor = rng.choice(timeframes)
    tf_trigger = rng.choice(timeframes)

    ef_lo, ef_hi = space["ema_fast"]
    es_lo, es_hi = space["ema_slow"]
    ef = _rand_int(rng, ef_lo, ef_hi)
    es = _rand_int(rng, max(es_lo, ef+1), es_hi)

    zlen = _rand_int(rng, *space["z_len"])
    zentry = 0.0 if zlen == 0 else _rand_float(rng, *space["z_entry"])

    extra_filters = []
    max_filters = int(space.get("max_extra_filters", 2))
    if reg and reg.list() and max_filters > 0:
        k = rng.randint(0, max_filters)
        for _ in range(k):
            extra_filters.append(_random_filter(rng, reg, space))

    return Genome(
        tf_anchor=tf_anchor,
        tf_trigger=tf_trigger,
        ema_fast=ef,
        ema_slow=es,
        rsi_len=_rand_int(rng, *space["rsi_len"]),
        rsi_long=_rand_float(rng, *space["rsi_long"]),
        rsi_short=_rand_float(rng, *space["rsi_short"]),
        atr_len=_rand_int(rng, *space["atr_len"]),
        sl_atr=_rand_float(rng, *space["sl_atr"]),
        tp_atr=_rand_float(rng, *space["tp_atr"]),
        z_len=zlen,
        z_entry=zentry,
        extra_filters=extra_filters,
    )

def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    def pick(x, y):
        return x if rng.random() < 0.5 else y
    child = Genome(
        tf_anchor=pick(a.tf_anchor, b.tf_anchor),
        tf_trigger=pick(a.tf_trigger, b.tf_trigger),
        ema_fast=pick(a.ema_fast, b.ema_fast),
        ema_slow=pick(a.ema_slow, b.ema_slow),
        rsi_len=pick(a.rsi_len, b.rsi_len),
        rsi_long=pick(a.rsi_long, b.rsi_long),
        rsi_short=pick(a.rsi_short, b.rsi_short),
        atr_len=pick(a.atr_len, b.atr_len),
        sl_atr=pick(a.sl_atr, b.sl_atr),
        tp_atr=pick(a.tp_atr, b.tp_atr),
        z_len=pick(a.z_len, b.z_len),
        z_entry=pick(a.z_entry, b.z_entry),
        extra_filters=list(a.extra_filters if rng.random()<0.5 else b.extra_filters),
    )
    if child.ema_fast >= child.ema_slow:
        child.ema_slow = child.ema_fast + 1
    if child.z_len == 0:
        child.z_entry = 0.0
    return child

def mutate(g: Genome, space: Dict[str, Any], rng: random.Random, rate: float, timeframes: List[str], reg: IndicatorRegistry | None) -> Genome:
    def m_int(val, lo, hi):
        if rng.random() < rate:
            step = rng.choice([-3, -2, -1, 1, 2, 3])
            return int(clip(int(val) + step, int(lo), int(hi)))
        return int(val)

    def m_float(val, lo, hi, scale=0.15):
        if rng.random() < rate:
            span = float(hi - lo)
            step = (rng.random() * 2 - 1) * span * scale
            return float(clip(float(val) + step, float(lo), float(hi)))
        return float(val)

    tf_anchor = g.tf_anchor if rng.random() >= rate else rng.choice(timeframes)
    tf_trigger = g.tf_trigger if rng.random() >= rate else rng.choice(timeframes)

    ema_fast = m_int(g.ema_fast, *space["ema_fast"])
    ema_slow = m_int(g.ema_slow, max(space["ema_slow"][0], ema_fast+1), space["ema_slow"][1])

    rsi_len = m_int(g.rsi_len, *space["rsi_len"])
    rsi_long = m_float(g.rsi_long, *space["rsi_long"])
    rsi_short = m_float(g.rsi_short, *space["rsi_short"])
    atr_len = m_int(g.atr_len, *space["atr_len"])
    sl_atr = m_float(g.sl_atr, *space["sl_atr"])
    tp_atr = m_float(g.tp_atr, *space["tp_atr"])

    z_len = m_int(g.z_len, *space["z_len"])
    z_entry = 0.0 if z_len == 0 else m_float(g.z_entry if g.z_len != 0 else 0.0, *space["z_entry"])

    # mutate filters: add/remove/modify threshold
    extra_filters = list(g.extra_filters or [])
    max_filters = int(space.get("max_extra_filters", 2))

    if reg and reg.list() and max_filters > 0 and rng.random() < rate:
        action = rng.choice(["add", "remove", "tweak", "replace"])
        if action == "add" and len(extra_filters) < max_filters:
            extra_filters.append(_random_filter(rng, reg, space))
        elif action == "remove" and extra_filters:
            extra_filters.pop(rng.randrange(len(extra_filters)))
        elif action == "replace" and extra_filters:
            extra_filters[rng.randrange(len(extra_filters))] = _random_filter(rng, reg, space)
        elif action == "tweak" and extra_filters:
            idx = rng.randrange(len(extra_filters))
            f = extra_filters[idx]
            # small tweak
            # Small tweaks with mode-awareness.
            # Also allow occasional mode switch when a valid scale is available.
            if reg is not None:
                try:
                    ind = reg.get(f.indicator_id)
                    out_cfg = ind.outputs.get(f.output, {}) if isinstance(ind.outputs, dict) else {}
                    scale = out_cfg.get("scale")
                    can_absolute = isinstance(scale, (list, tuple)) and len(scale) == 2 and scale[0] is not None and scale[1] is not None
                except Exception:
                    can_absolute = False
                    scale = None
            else:
                can_absolute = False
                scale = None

            # Optional mode switch
            if rng.random() < 0.10 and can_absolute:
                new_mode = "absolute" if f.mode == "quantile" else "quantile"
            else:
                new_mode = f.mode

            if new_mode == "absolute" and can_absolute:
                lo, hi = float(scale[0]), float(scale[1])
                if lo > hi:
                    lo, hi = hi, lo
                # tweak threshold within [lo, hi]
                thr = float(f.threshold)
                if thr == 0.0:
                    thr = float(lo + (hi - lo) * rng.random())
                else:
                    thr = float(thr * (1.0 + (rng.random() * 2 - 1) * 0.15))
                thr = float(clip(thr, lo, hi))
                extra_filters[idx] = IndicatorFilter(
                    indicator_id=f.indicator_id,
                    output=f.output,
                    comparator=f.comparator,
                    timeframe=f.timeframe,
                    mode="absolute",
                    threshold=thr,
                    quantile=f.quantile,
                    lookback=f.lookback,
                )
            else:
                # quantile tweak
                q_lo, q_hi = 0.1, 0.9
                lb_lo, lb_hi = 200, 1500
                if space:
                    if "filter_quantile" in space:
                        q_lo, q_hi = float(space["filter_quantile"][0]), float(space["filter_quantile"][1])
                    if "filter_lookback" in space:
                        lb_lo, lb_hi = int(space["filter_lookback"][0]), int(space["filter_lookback"][1])
                q = float(f.quantile + (rng.random() * 2 - 1) * 0.05)
                q = float(clip(q, q_lo, q_hi))
                lb = int(f.lookback + rng.choice([-150, -100, -50, 50, 100, 150]))
                lb = int(clip(lb, lb_lo, lb_hi))
                extra_filters[idx] = IndicatorFilter(
                    indicator_id=f.indicator_id,
                    output=f.output,
                    comparator=f.comparator,
                    timeframe=f.timeframe,
                    mode="quantile",
                    threshold=f.threshold,
                    quantile=q,
                    lookback=lb,
                )

    out = Genome(tf_anchor, tf_trigger, ema_fast, ema_slow, rsi_len, rsi_long, rsi_short, atr_len, sl_atr, tp_atr, z_len, z_entry, extra_filters)
    if out.ema_fast >= out.ema_slow:
        out.ema_slow = out.ema_fast + 1
    if out.z_len == 0:
        out.z_entry = 0.0
    return out
