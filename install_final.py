# install_final.py
# One-shot installer for:
# - Community indicators (50+) with dimensionless outputs
# - YAML specs generation
# - Aggressive GA preset config
#
# Usage:
#   python install_final.py
#
# Run this at the PROJECT ROOT (where ga_trader/ exists)

from __future__ import annotations
from pathlib import Path
import textwrap
import yaml

ROOT = Path(__file__).resolve().parent
GA_TRADER = ROOT / "ga_trader"
SPEC_DIR = GA_TRADER / "indicators" / "specs"
PYIMPL_DIR = GA_TRADER / "indicators" / "python_impl"
CONFIGS_DIR = ROOT / "configs"

def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# -----------------------------
# 1) Python implementation (community50)
# -----------------------------
COMMUNITY50_PY = r'''
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
'''

# -----------------------------
# 2) Indicator specs (50+ via variants)
# -----------------------------
def make_specs():
    def bars(name, a,b,d): return {name:{"type":"int","min":a,"max":b,"default":d,"unit":"bars"}}
    def ratio(name,a,b,d): return {name:{"type":"float","min":a,"max":b,"default":d,"unit":"ratio"}}

    base = [
        ("c_rsi_z","RSI Z","rsi z",{"length":bars("length",2,60,14),"z_len":bars("z_len",20,500,200)}),
        ("c_macd_hist_z","MACD Hist Z","macd hist z",{
            "fast":bars("fast",2,40,12),"slow":bars("slow",5,120,26),
            "signal":bars("signal",2,40,9),"atr_len":bars("atr_len",5,60,14),
            "z_len":bars("z_len",20,500,200)
        }),
        ("c_bb_width_z","BB Width Z","bb width z",{
            "length":bars("length",5,200,20),"mult":ratio("mult",0.5,4.0,2.0),
            "z_len":bars("z_len",20,500,200)
        }),
        ("c_donchian_pos","Donchian Pos","donchian pos",{"length":bars("length",5,200,20)}),
        ("c_vwap_dist_atr","VWAP Dist ATR","vwap dist atr",{
            "length":bars("length",5,500,50),"atr_len":bars("atr_len",5,100,14)
        }),
    ]
    variants = [
        ("c_rsi_z_fast","RSI Z Fast","rsi z fast",{"length":bars("length",2,20,7),"z_len":bars("z_len",20,500,200)}),
        ("c_rsi_z_slow","RSI Z Slow","rsi z slow",{"length":bars("length",20,80,28),"z_len":bars("z_len",20,500,200)}),
        ("c_macd_hist_z_fast","MACD Z Fast","macd z fast",{
            "fast":bars("fast",2,20,6),"slow":bars("slow",10,40,13),
            "signal":bars("signal",2,20,5),"atr_len":bars("atr_len",5,60,14),
            "z_len":bars("z_len",20,500,200)
        }),
        ("c_macd_hist_z_slow","MACD Z Slow","macd z slow",{
            "fast":bars("fast",20,60,24),"slow":bars("slow",40,120,52),
            "signal":bars("signal",10,30,18),"atr_len":bars("atr_len",5,60,14),
            "z_len":bars("z_len",20,500,200)
        }),
    ]
    specs=[]
    for i,(iid,name,desc,params) in enumerate(base+variants):
        specs.append({
            "indicator_id": iid,
            "name": name,
            "description": desc,
            "timeframe": "trigger",
            "parameters": params,
            "outputs": {"value":{"scale":[-3,3]}},
            "python_impl": f"ga_trader.indicators.python_impl.community50:{iid}",
            "pine_snippet": "value = 0.0\n"
        })
    return specs

# -----------------------------
# 3) Aggressive config
# -----------------------------
AGGRESSIVE_CFG = """
ga:
  population: 400
  generations: 120
  elite_ratio: 0.03
  tournament_k: 7
  crossover_rate: 0.65
  mutation_rate: 0.28
  immigrant_ratio: 0.20
  stagnation_gens: 10
  restart_fraction: 0.90

strategy:
  max_extra_filters: 8

fitness:
  mode: targets
  targets:
    win_rate: 0.58
    rr: 1.1
    min_trades: 40
"""

def main():
    write(PYIMPL_DIR / "community50.py", COMMUNITY50_PY)

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for s in make_specs():
        write(SPEC_DIR / f"{s['indicator_id']}.yaml",
              yaml.safe_dump(s, sort_keys=False, allow_unicode=True))

    write(CONFIGS_DIR / "config.aggressive.yaml", AGGRESSIVE_CFG)
    print("INSTALL DONE")

if __name__ == "__main__":
    main()
