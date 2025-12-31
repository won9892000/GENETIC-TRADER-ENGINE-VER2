from __future__ import annotations
import numpy as np
from ga_trader.ta.core import sma, stdev, atr

def bbwidth(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, length: int, mult: float):
    basis = sma(close, int(length))
    dev = float(mult) * stdev(close, int(length))
    upper = basis + dev
    lower = basis - dev
    width_pct = (upper - lower) / basis
    return {"width_pct": width_pct}

def atrpct(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, length: int):
    a = atr(high, low, close, int(length))
    atr_pct = a / close
    return {"atr_pct": atr_pct}
