from __future__ import annotations
from typing import List, Dict, Any
import time
import requests
import pandas as pd


def fetch_binance_usdt_perp_symbols(base_url: str = "https://fapi.binance.com") -> List[Dict[str, Any]]:
    """Fetch Binance USDT-M perpetual futures symbols and basic metadata."""
    url = base_url.rstrip("/") + "/fapi/v1/exchangeInfo"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: List[Dict[str, Any]] = []

    for s in data.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue

        tick_size = None
        step_size = None
        for f in s.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                try:
                    tick_size = float(f.get("tickSize"))
                except Exception:
                    pass
            if f.get("filterType") == "LOT_SIZE":
                try:
                    step_size = float(f.get("stepSize"))
                except Exception:
                    pass

        out.append({
            "symbol": s.get("symbol"),
            "baseAsset": s.get("baseAsset"),
            "quoteAsset": s.get("quoteAsset"),
            "tick_size": tick_size,
            "step_size": step_size,
        })
    return out


def fetch_top_volume_usdt_perp_symbols(
    base_url: str = "https://fapi.binance.com",
    topn: int = 80,
) -> List[Dict[str, Any]]:
    """Return USDT-margined perpetual symbols sorted by 24h quote volume.

    Output items include:
      - symbol
      - baseAsset
      - quoteAsset
      - tick_size (from PRICE_FILTER tickSize)
      - step_size (from LOT_SIZE stepSize)
      - quote_volume_24h (from /fapi/v1/ticker/24hr quoteVolume)
    """
    base_url = base_url.rstrip("/")
    t_url = base_url + "/fapi/v1/ticker/24hr"
    r = requests.get(t_url, timeout=30)
    r.raise_for_status()
    tickers = r.json()

    meta_rows = fetch_binance_usdt_perp_symbols(base_url)
    meta = {m["symbol"]: m for m in meta_rows}

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or sym not in meta:
            continue
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        row = dict(meta[sym])
        row["quote_volume_24h"] = qv
        rows.append(row)

    rows.sort(key=lambda x: float(x.get("quote_volume_24h", 0.0)), reverse=True)
    return rows[:int(topn)]

def fetch_klines(
    base_url: str,
    symbol: str,
    interval: str,
    lookback_days: int = 180,
    sleep_sec: float = 0.15,
) -> pd.DataFrame:
    '''
    Fetch klines from Binance USDT-M Futures.
    Note: Binance limits 1500 bars per call. We page backwards from now using endTime.
    '''
    endpoint = "/fapi/v1/klines"
    url = base_url.rstrip("/") + endpoint
    limit = 1500
    ms_per_bar = {
        "1m": 60_000,
        "3m": 3*60_000,
        "5m": 5*60_000,
        "15m": 15*60_000,
    }.get(interval)
    if ms_per_bar is None:
        raise ValueError(f"Unsupported interval: {interval}")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(lookback_days) * 24 * 60 * 60 * 1000

    rows = []
    end = now_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        earliest = int(data[0][0])
        if earliest <= start_ms:
            break
        # move end earlier
        end = earliest - 1
        time.sleep(sleep_sec)

    # Binance returns newest-last within page; combined pages are reverse chronological
    rows = sorted(rows, key=lambda x: x[0])
    # trim to start_ms
    rows = [x for x in rows if int(x[0]) >= start_ms]

    df = pd.DataFrame(rows, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df



def fetch_top_volume_usdt_perp_symbols(
    base_url: str = "https://fapi.binance.com",
    topn: int = 80,
) -> List[Dict[str, Any]]:
    """Return USDT-margined perpetual symbols sorted by 24h quote volume.

    Output items include:
      - symbol
      - baseAsset
      - quoteAsset
      - tick_size (from PRICE_FILTER tickSize)
      - step_size (from LOT_SIZE stepSize)
      - quote_volume_24h (from /fapi/v1/ticker/24hr quoteVolume)
    """
    base_url = base_url.rstrip("/")
    # 24h ticker: includes quoteVolume and volume
    t_url = base_url + "/fapi/v1/ticker/24hr"
    r = requests.get(t_url, timeout=30)
    r.raise_for_status()
    tickers = r.json()

    # exchangeInfo for metadata and tick size
    info = fetch_binance_usdt_perp_symbols(base_url)
    meta = {i["symbol"]: i for i in info}

    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or sym not in meta:
            continue
        # keep only PERPETUAL USDT (meta already filtered)
        try:
            qv = float(t.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        row = dict(meta[sym])
        row["quote_volume_24h"] = qv
        rows.append(row)

    rows.sort(key=lambda x: float(x.get("quote_volume_24h", 0.0)), reverse=True)
    return rows[:int(topn)]
