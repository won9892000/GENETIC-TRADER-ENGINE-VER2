from __future__ import annotations
import numpy as np
import pandas as pd
import time

def make_synthetic_ohlcv(n: int = 3000, start_ts_ms: int | None = None, minutes: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if start_ts_ms is None:
        start_ts_ms = int(time.time() * 1000) - n * minutes * 60_000
    ts = start_ts_ms + np.arange(n) * minutes * 60_000

    # random walk with volatility regimes
    vol = np.where((np.arange(n) // 300) % 2 == 0, 0.002, 0.006)
    rets = rng.normal(0, vol)
    price = 100 * np.exp(np.cumsum(rets))
    close = price
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.002)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.002)
    volume = rng.lognormal(mean=2.0, sigma=0.4, size=n)

    return pd.DataFrame({
        "timestamp": ts.astype("int64"),
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": volume.astype(float),
    })
