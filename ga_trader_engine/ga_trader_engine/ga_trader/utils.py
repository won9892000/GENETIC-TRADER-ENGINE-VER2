from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json
import time
import math
import numpy as np
import pandas as pd

def now_utc_compact() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")

def bps_to_rate(bps: float) -> float:
    return float(bps) / 10000.0

def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b if abs(b) > eps else eps)

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def month_bucket(ts_ms: int) -> str:
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts_ms/1000)
    return f"{dt.year:04d}-{dt.month:02d}"


def round_to_tick(price: float, tick_size: float, mode: str = "nearest") -> float:
    """Round price to exchange tick size.
    mode: nearest|floor|ceil
    """
    if tick_size is None or tick_size <= 0:
        return float(price)
    x = float(price) / float(tick_size)
    if mode == "floor":
        y = math.floor(x)
    elif mode == "ceil":
        y = math.ceil(x)
    else:
        y = round(x)
    return float(y * float(tick_size))

def apply_slippage_ticks(price: float, tick_size: float, ticks: int, side: str) -> float:
    """Apply slippage in ticks.
    side: 'buy' (worse higher) or 'sell' (worse lower)
    """
    if ticks is None:
        ticks = 0
    if tick_size is None or tick_size <= 0:
        # fallback: treat 1 tick as 0
        return float(price)
    adj = float(ticks) * float(tick_size)
    if side == "buy":
        return float(price + adj)
    elif side == "sell":
        return float(price - adj)
    else:
        return float(price)
