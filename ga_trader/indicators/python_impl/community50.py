
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12

def _ema(x, n): return x.ewm(span=max(1,int(n)), adjust=False).mean()
def _sma(x, n): return x.rolling(max(1,int(n))).mean()
def _std(x, n): return x.rolling(max(1,int(n))).std(ddof=0)
def _z(x, n): return (x - _sma(x,n)) / (_std(x,n) + EPS)

def _tr(df):
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def _atr(df, n): return _ema(_tr(df), n)

def c_rsi_z(df, length=14, z_len=200):
    d = df["close"].diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = _ema(up,length) / (_ema(dn,length) + EPS)
    rsi = 100 - 100/(1+rs)
    return {"value": _z(rsi, z_len)}

def c_macd_hist_z(df, fast=12, slow=26, signal=9, atr_len=14, z_len=200):
    c = df["close"]
    macd = _ema(c,fast) - _ema(c,slow)
    hist = macd - _ema(macd,signal)
    hist_n = hist / (_atr(df,atr_len) + EPS)
    return {"value": _z(hist_n, z_len)}

def c_bb_width_z(df, length=20, mult=2.0, z_len=200):
    c = df["close"]
    basis = _sma(c,length)
    dev = mult * _std(c,length)
    width = (basis+dev - (basis-dev)) / (basis.abs()+EPS)
    return {"value": _z(width, z_len)}

def c_donchian_pos(df, length=20):
    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()
    pos = (df["close"]-ll)/((hh-ll)+EPS)
    return {"value": pos*2-1}

def c_vwap_dist_atr(df, length=50, atr_len=14):
    tp = (df["high"]+df["low"]+df["close"])/3
    vwap = (tp*df["volume"]).rolling(length).sum()/(df["volume"].rolling(length).sum()+EPS)
    return {"value": (df["close"]-vwap)/(_atr(df,atr_len)+EPS)}

# variants to expand pool
def c_rsi_z_fast(df, length=7, z_len=200): return c_rsi_z(df,length,z_len)
def c_rsi_z_slow(df, length=28, z_len=200): return c_rsi_z(df,length,z_len)
def c_macd_hist_z_fast(df, fast=6, slow=13, signal=5, atr_len=14, z_len=200):
    return c_macd_hist_z(df,fast,slow,signal,atr_len,z_len)
def c_macd_hist_z_slow(df, fast=24, slow=52, signal=18, atr_len=14, z_len=200):
    return c_macd_hist_z(df,fast,slow,signal,atr_len,z_len)
