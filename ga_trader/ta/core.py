from __future__ import annotations
import numpy as np
import pandas as pd


def sma(x: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be >0")
    n = len(x)
    out = np.full_like(x, np.nan, dtype=float)
    if n == 0:
        return out
    x0 = np.nan_to_num(x, nan=0.0)
    cs = np.cumsum(x0)
    cn = np.cumsum(~np.isnan(x)).astype(float)

    i = np.arange(n)
    j = i - length + 1
    valid = j >= 0
    # sum and count for window ending at i
    sum_w = cs - np.where(j > 0, cs[j - 1], 0.0)
    count = cn - np.where(j > 0, cn[j - 1], 0.0)
    mask = valid & (count > 0)
    out[mask] = sum_w[mask] / count[mask]
    return out

def ema(x: np.ndarray, length: int) -> np.ndarray:
    # Pine ta.ema: seed with sma over first length bars (first non-nan window)
    if length <= 0:
        raise ValueError("length must be >0")
    out = np.full_like(x, np.nan, dtype=float)
    alpha = 2.0 / (length + 1.0)

    # find first index where we have length non-nan values in a row
    vals = x.astype(float)
    n = len(vals)
    # Build rolling window by scanning; robust to leading NaNs
    valid_idx = np.where(~np.isnan(vals))[0]
    if len(valid_idx) == 0:
        return out
    # compute seed SMA at the earliest point where we have length valid points cumulatively (not necessarily contiguous)
    # Pine effectively uses the first length bars of series; if NaNs exist, behavior is tricky.
    # We enforce: require first length non-NaN values (in order) to seed.
    if len(valid_idx) < length:
        return out
    seed_end = valid_idx[length-1]
    seed_start = valid_idx[0]
    # seed uses the first length non-nan values, not a time window. This approximates Pine for typical OHLC (no NaNs).
    seed_vals = vals[valid_idx[:length]]
    seed = float(np.mean(seed_vals))
    out[seed_end] = seed

    for i in range(seed_end + 1, n):
        if np.isnan(vals[i]):
            out[i] = out[i-1]
        else:
            out[i] = alpha * vals[i] + (1 - alpha) * out[i-1]
    return out

def rma(x: np.ndarray, length: int) -> np.ndarray:
    # Pine ta.rma (Wilder): seed with sma of first length values, then:
    # rma[i] = (rma[i-1]*(length-1) + x[i]) / length
    if length <= 0:
        raise ValueError("length must be >0")
    out = np.full_like(x, np.nan, dtype=float)
    vals = x.astype(float)
    valid_idx = np.where(~np.isnan(vals))[0]
    if len(valid_idx) < length:
        return out
    seed_end = valid_idx[length-1]
    seed = float(np.mean(vals[valid_idx[:length]]))
    out[seed_end] = seed
    for i in range(seed_end + 1, len(vals)):
        if np.isnan(vals[i]):
            out[i] = out[i-1]
        else:
            out[i] = (out[i-1] * (length - 1) + vals[i]) / length
    return out

def rsi(close: np.ndarray, length: int) -> np.ndarray:
    # Pine ta.rsi uses RMA of gains/losses
    close = close.astype(float)
    delta = np.full_like(close, np.nan, dtype=float)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=~np.isnan(avg_gain) & ~np.isnan(avg_loss) & (avg_loss != 0))
    out = 100.0 - (100.0 / (1.0 + rs))
    # handle avg_loss == 0 => rsi 100, avg_gain==0 => rsi 0; both 0 => 50
    both = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss))
    out = np.where(both & (avg_loss == 0) & (avg_gain == 0), 50.0, out)
    out = np.where(both & (avg_loss == 0) & (avg_gain != 0), 100.0, out)
    out = np.where(both & (avg_gain == 0) & (avg_loss != 0), 0.0, out)
    return out

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = np.full_like(close, np.nan, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    return tr

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = true_range(high, low, close)
    return rma(tr, length)

def stdev(x: np.ndarray, length: int) -> np.ndarray:
    # Pine ta.stdev uses population stdev by default (ddof=0)
    if length <= 0:
        raise ValueError("length must be >0")
    n = len(x)
    out = np.full_like(x, np.nan, dtype=float)
    if n == 0:
        return out
    x0 = np.nan_to_num(x, nan=0.0)
    cs = np.cumsum(x0)
    cs2 = np.cumsum(x0 * x0)
    cn = np.cumsum(~np.isnan(x)).astype(float)

    i = np.arange(n)
    j = i - length + 1
    valid = j >= 0
    sum_w = cs - np.where(j > 0, cs[j - 1], 0.0)
    sumsq_w = cs2 - np.where(j > 0, cs2[j - 1], 0.0)
    count = cn - np.where(j > 0, cn[j - 1], 0.0)

    mask = valid & (count > 0)
    # var = E[x^2] - (E[x])^2
    mean = np.zeros_like(out)
    mean[mask] = sum_w[mask] / count[mask]
    var = np.zeros_like(out)
    var[mask] = (sumsq_w[mask] / count[mask]) - (mean[mask] ** 2)
    var = np.where(var < 0, 0.0, var)
    out[mask] = np.sqrt(var[mask])
    return out

def zscore(x: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.full_like(x, np.nan, dtype=float)
    mu = sma(x, length)
    sd = stdev(x, length)
    out = (x - mu) / sd
    out = np.where(sd == 0, np.nan, out)
    return out


def percentile_linear_interpolation(x: np.ndarray, length: int, percentile: float) -> np.ndarray:
    """Approximate Pine ta.percentile_linear_interpolation.
    percentile: 0..100
    """
    if length <= 0:
        raise ValueError("length must be >0")
    p = float(percentile) / 100.0
    s = pd.Series(x)
    try:
        out = s.rolling(window=length, min_periods=length).quantile(p, interpolation='linear').to_numpy(dtype=float)
    except TypeError:
        # pandas older versions use method= instead of interpolation=
        out = s.rolling(window=length, min_periods=length).quantile(p, method='linear').to_numpy(dtype=float)
    return out