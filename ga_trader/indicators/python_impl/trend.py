from __future__ import annotations
import numpy as np
from ga_trader.ta.core import atr

def _vwma(close: np.ndarray, volume: np.ndarray, length: int) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=float)
    length = int(length)
    for i in range(len(close)):
        j = i - length + 1
        if j < 0:
            continue
        c = close[j:i+1]
        v = volume[j:i+1]
        mask = ~np.isnan(c) & ~np.isnan(v)
        if mask.sum() == 0:
            continue
        num = float(np.sum(c[mask] * v[mask]))
        den = float(np.sum(v[mask]))
        if den == 0:
            continue
        out[i] = num / den
    return out

def vwma_dist_atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, length: int):
    vw = _vwma(close, volume, int(length))
    a = atr(high, low, close, 14)
    dist_atr = np.abs(close - vw) / a
    return {"dist_atr": dist_atr}
