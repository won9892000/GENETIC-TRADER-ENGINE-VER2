from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12

def _to_series(arr):
    if isinstance(arr, pd.Series):
        return arr
    return pd.Series(arr)

def _ema(x, n):
    s = _to_series(x)
    return s.ewm(span=max(1,int(n)), adjust=False).mean()

def _sma(x, n):
    s = _to_series(x)
    return s.rolling(max(1,int(n))).mean()

def _std(x, n):
    s = _to_series(x)
    return s.rolling(max(1,int(n))).std(ddof=0)

def _z(x, n):
    s = _to_series(x)
    return (s - _sma(s,n)) / (_std(s,n) + EPS)

def _tr(high, low, close):
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def _atr(high, low, close, n):
    return _ema(_tr(high, low, close), n)

def c_rsi_z(close, high=None, low=None, volume=None, length=14, z_len=200, **kwargs):
    c = _to_series(close)
    d = c.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = _ema(up, length) / (_ema(dn, length) + EPS)
    rsi = 100 - 100/(1+rs)
    return {"value": _z(rsi, z_len)}

def c_macd_hist_z(close, high=None, low=None, volume=None, fast=12, slow=26, signal=9, atr_len=14, z_len=200, **kwargs):
    c = _to_series(close)
    h = _to_series(high) if high is not None else c
    l = _to_series(low) if low is not None else c
    macd = _ema(c, fast) - _ema(c, slow)
    hist = macd - _ema(macd, signal)
    hist_n = hist / (_atr(h, l, c, atr_len) + EPS)
    return {"value": _z(hist_n, z_len)}

def c_bb_width_z(close, high=None, low=None, volume=None, length=20, mult=2.0, z_len=200, **kwargs):
    c = _to_series(close)
    basis = _sma(c, length)
    dev = mult * _std(c, length)
    width = (basis+dev - (basis-dev)) / (basis.abs()+EPS)
    return {"value": _z(width, z_len)}

def c_donchian_pos(close, high=None, low=None, volume=None, length=20, **kwargs):
    c = _to_series(close)
    h = _to_series(high) if high is not None else c
    l = _to_series(low) if low is not None else c
    hh = h.rolling(length).max()
    ll = l.rolling(length).min()
    pos = (c - ll) / ((hh - ll) + EPS)
    return {"value": pos*2-1}

def c_vwap_dist_atr(close, high=None, low=None, volume=None, length=50, atr_len=14, **kwargs):
    c = _to_series(close)
    h = _to_series(high) if high is not None else c
    l = _to_series(low) if low is not None else c
    v = _to_series(volume) if volume is not None else pd.Series(np.ones(len(c)))
    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(length).sum() / (v.rolling(length).sum() + EPS)
    return {"value": (c - vwap) / (_atr(h, l, c, atr_len) + EPS)}

def c_rsi_z_fast(close, high=None, low=None, volume=None, length=7, z_len=200, **kwargs):
    return c_rsi_z(close, high, low, volume, length, z_len, **kwargs)

def c_rsi_z_slow(close, high=None, low=None, volume=None, length=28, z_len=200, **kwargs):
    return c_rsi_z(close, high, low, volume, length, z_len, **kwargs)

def c_macd_hist_z_fast(close, high=None, low=None, volume=None, fast=6, slow=13, signal=5, atr_len=14, z_len=200, **kwargs):
    return c_macd_hist_z(close, high, low, volume, fast, slow, signal, atr_len, z_len, **kwargs)

def c_macd_hist_z_slow(close, high=None, low=None, volume=None, fast=24, slow=52, signal=18, atr_len=14, z_len=200, **kwargs):
    return c_macd_hist_z(close, high, low, volume, fast, slow, signal, atr_len, z_len, **kwargs)
