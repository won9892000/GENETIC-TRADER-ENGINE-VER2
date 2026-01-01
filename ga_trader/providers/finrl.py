"""Optional FinRL / TA adapter: provide many technical indicators and helpers

This module does NOT require finrl to be installed. If FinRL is available, certain helpers will use it.
We provide compute_* functions that match the indicator python_impl contract used by IndicatorSpec:
    def f(close, high, low, volume, **kwargs) -> Dict[str, np.ndarray]

We also provide a helper to generate YAML IndicatorSpec drafts under `indicators/specs/`.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import numpy as np

# optional imports
try:
    import finrl
    _FINRL_AVAILABLE = True
except Exception:
    _FINRL_AVAILABLE = False

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except Exception:
    _YFINANCE_AVAILABLE = False

# prefer built-in ta functions where available
from ga_trader.ta.core import ema, rsi, atr

# helpers

def _safe_series(arr):
    return np.asarray(arr, dtype=float)

# Indicator implementations

def compute_rsi(close, high=None, low=None, volume=None, length=14, **kwargs) -> Dict[str, np.ndarray]:
    out = rsi(np.asarray(close, dtype=float), int(length))
    return {"rsi": _safe_series(out)}


def compute_ema(close, high=None, low=None, volume=None, length=14, **kwargs) -> Dict[str, np.ndarray]:
    out = ema(np.asarray(close, dtype=float), int(length))
    return {f"ema_{int(length)}": _safe_series(out)}


def compute_atr(close, high, low, volume=None, length=14, **kwargs) -> Dict[str, np.ndarray]:
    out = atr(np.asarray(high, dtype=float), np.asarray(low, dtype=float), np.asarray(close, dtype=float), int(length))
    return {"atr": _safe_series(out)}


def compute_macd(close, high=None, low=None, volume=None, fast=12, slow=26, signal=9, **kwargs) -> Dict[str, np.ndarray]:
    c = np.asarray(close, dtype=float)
    fast_ema = ema(c, int(fast))
    slow_ema = ema(c, int(slow))
    macd = fast_ema - slow_ema
    sig = ema(macd, int(signal))
    hist = macd - sig
    return {"macd": _safe_series(macd), "macd_signal": _safe_series(sig), "macd_hist": _safe_series(hist)}


def compute_bbands(close, high=None, low=None, volume=None, length=20, mult=2.0, **kwargs) -> Dict[str, np.ndarray]:
    c = np.asarray(close, dtype=float)
    n = len(c)
    out_mid = np.full(n, np.nan)
    out_up = np.full(n, np.nan)
    out_lo = np.full(n, np.nan)
    import numpy as _np
    for i in range(n):
        j = i - int(length) + 1
        if j < 0:
            continue
        w = c[j:i+1]
        w = w[~_np.isnan(w)]
        if len(w) == 0:
            continue
        m = float(_np.mean(w))
        s = float(_np.std(w, ddof=0))
        out_mid[i] = m
        out_up[i] = m + mult * s
        out_lo[i] = m - mult * s
    return {"bb_mid": out_mid, "bb_upper": out_up, "bb_lower": out_lo}


def compute_obv(close, high=None, low=None, volume=None, **kwargs) -> Dict[str, np.ndarray]:
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    out = np.zeros_like(c)
    cur = 0.0
    for i in range(len(c)):
        if i == 0:
            out[i] = 0.0
            continue
        if np.isnan(c[i]) or np.isnan(c[i-1]):
            out[i] = out[i-1]
            continue
        if c[i] > c[i-1]:
            cur += v[i]
        elif c[i] < c[i-1]:
            cur -= v[i]
        out[i] = cur
    return {"obv": out}


def compute_roc(close, high=None, low=None, volume=None, length=12, **kwargs) -> Dict[str, np.ndarray]:
    c = np.asarray(close, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n):
        j = i - int(length)
        if j < 0:
            continue
        if np.isnan(c[j]) or np.isnan(c[i]) or c[j] == 0:
            continue
        out[i] = (c[i] - c[j]) / c[j]
    return {"roc": out}


def compute_cci(close, high, low, volume=None, length=20, **kwargs) -> Dict[str, np.ndarray]:
    c = np.asarray(close, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    tp = (h + l + c) / 3.0
    n = len(tp)
    sma = np.full(n, np.nan)
    for i in range(n):
        j = i - int(length) + 1
        if j < 0:
            continue
        w = tp[j:i+1]
        w = w[~np.isnan(w)]
        if len(w) == 0:
            continue
        sma[i] = float(np.mean(w))
    md = np.full(n, np.nan)
    for i in range(n):
        j = i - int(length) + 1
        if j < 0 or np.isnan(sma[i]):
            continue
        w = tp[j:i+1]
        w = w[~np.isnan(w)]
        md[i] = float(np.mean(np.abs(w - sma[i]))) if len(w) else np.nan
    cci = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(sma[i]) or np.isnan(md[i]) or md[i] == 0:
            continue
        cci[i] = (tp[i] - sma[i]) / (0.015 * md[i])
    return {"cci": cci}


def compute_indicators_bulk(close, high, low, volume, **kwargs) -> Dict[str, np.ndarray]:
    """Return a dictionary with a broad set of indicators (useful for generating specs).
    WARNING: returns many arrays; callers may choose a subset."""
    out = {}
    out.update(compute_rsi(close, length=14))
    out.update(compute_ema(close, length=12))
    out.update(compute_ema(close, length=26))
    out.update(compute_macd(close, fast=12, slow=26, signal=9))
    out.update(compute_bbands(close, length=20, mult=2.0))
    out.update(compute_obv(close, volume=volume))
    out.update(compute_roc(close, length=12))
    out.update(compute_cci(close, high=high, low=low, length=20))
    # more can be added
    return out

# YAML spec generation helper

def generate_indicator_specs(root: Path | str = "indicators") -> int:
    """Generate simple spec YAML files for a core set of FinRL-style indicators.
    Returns number of specs written."""
    import yaml
    rootp = Path(root)
    specs_dir = rootp / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {"indicator_id": "finrl_rsi", "name": "FinRL RSI", "timeframe": "trigger", "parameters": {"length": {"type": "int", "min": 2, "max": 100, "default": 14}}, "outputs": {"rsi": {"kind": "value", "scale": [0.0, 100.0]}}, "python_impl": "ga_trader.providers.finrl:compute_rsi", "pine_snippet": ""},
        {"indicator_id": "finrl_macd", "name": "FinRL MACD", "timeframe": "trigger", "parameters": {"fast": {"type": "int", "min": 2, "max": 50, "default": 12}, "slow": {"type": "int", "min": 10, "max": 200, "default": 26}, "signal": {"type": "int", "min": 2, "max": 50, "default": 9}}, "outputs": {"macd": {"kind": "value"}, "macd_signal": {"kind": "value"}, "macd_hist": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_macd", "pine_snippet": ""},
        {"indicator_id": "finrl_bbands", "name": "FinRL Bollinger Bands", "timeframe": "trigger", "parameters": {"length": {"type": "int", "min": 2, "max": 200, "default": 20}, "mult": {"type": "float", "min": 0.5, "max": 4.0, "default": 2.0}}, "outputs": {"bb_mid": {"kind": "value"}, "bb_upper": {"kind": "value"}, "bb_lower": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_bbands", "pine_snippet": ""},
        {"indicator_id": "finrl_atr", "name": "FinRL ATR", "timeframe": "anchor", "parameters": {"length": {"type": "int", "min": 2, "max": 200, "default": 14}}, "outputs": {"atr": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_atr", "pine_snippet": ""},
        {"indicator_id": "finrl_obv", "name": "FinRL OBV", "timeframe": "trigger", "parameters": {}, "outputs": {"obv": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_obv", "pine_snippet": ""},
        {"indicator_id": "finrl_roc", "name": "FinRL ROC", "timeframe": "trigger", "parameters": {"length": {"type": "int", "min": 2, "max": 200, "default": 12}}, "outputs": {"roc": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_roc", "pine_snippet": ""},
        {"indicator_id": "finrl_cci", "name": "FinRL CCI", "timeframe": "trigger", "parameters": {"length": {"type": "int", "min": 2, "max": 200, "default": 20}}, "outputs": {"cci": {"kind": "value"}}, "python_impl": "ga_trader.providers.finrl:compute_cci", "pine_snippet": ""},
    ]

    written = 0
    for s in specs:
        p = specs_dir / f"{s['indicator_id']}.yaml"
        p.write_text(yaml.safe_dump(s, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written += 1
    return written

# Simple data fetch using yfinance (optional)

def fetch_price_yf(symbol: str, period: str = "180d", interval: str = "5m"):
    if not _YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed; install with .[finrl] to use FinRL data helpers")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data from yfinance for {symbol} interval={interval} period={period}")
    # normalize to expected columns
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.reset_index()
    df["timestamp"] = (df["Datetime"].astype('int64') // 1_000_000) if "Datetime" in df.columns else (df.index.astype('int64') // 1_000_000)
    # If Datetime column exists, drop it
    df = df.drop(columns=[c for c in ["Datetime"] if c in df.columns])
    return df
