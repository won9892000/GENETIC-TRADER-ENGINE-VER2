from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import importlib

from ga_trader.ta.core import ema, rsi, atr, zscore, percentile_linear_interpolation
from ga_trader.utils import bps_to_rate, safe_div, month_bucket, round_to_tick, apply_slippage_ticks
from ga_trader.indicators.registry import IndicatorRegistry
from ga_trader.strategy.spec import StrategySpec, IndicatorFilter

@dataclass
class Trade:
    symbol: str
    timeframe: str
    entry_ts: int
    exit_ts: int
    side: str  # 'long'|'short'
    entry_price: float
    exit_price: float
    pnl: float
    pnl_r: float
    fee: float

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: np.ndarray
    metrics: Dict[str, float]

def _compute_metrics(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "avg_r": 0.0,
            "trades_per_month": 0.0,
            "net_pnl": 0.0,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = -pnls[pnls < 0]
    win_rate = float(np.mean(pnls > 0))
    pf = float(np.sum(wins) / np.sum(losses)) if np.sum(losses) > 0 else float("inf")
    exp = float(np.mean(pnls))
    rs = np.array([t.pnl_r for t in trades], dtype=float)
    avg_r = float(np.nanmean(rs)) if len(rs) else 0.0

    buckets = {}
    for t in trades:
        b = month_bucket(t.entry_ts)
        buckets[b] = buckets.get(b, 0) + 1
    tpm = float(np.mean(list(buckets.values()))) if buckets else 0.0

    eq = np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    mdd = float(np.max(dd)) if len(dd) else 0.0
    denom = float(np.max(np.abs(peak))) if float(np.max(np.abs(peak))) > 0 else float(np.max(np.abs(eq))) if len(eq) else 1.0
    mdd_ratio = float(mdd / denom) if denom > 0 else 0.0

    return {
        "trades": float(len(trades)),
        "win_rate": win_rate,
        "profit_factor": pf if np.isfinite(pf) else 999.0,
        "expectancy": exp,
        "max_drawdown": mdd_ratio,
        "avg_r": avg_r,
        "trades_per_month": tpm,
        "net_pnl": float(np.sum(pnls)),
    }

def _align_anchor_to_trigger(anchor_df: pd.DataFrame, trigger_df: pd.DataFrame, series: np.ndarray) -> np.ndarray:
    # Vectorized implementation using searchsorted: for each trigger ts pick the last anchor ts <= trigger ts
    a_ts = anchor_df["timestamp"].to_numpy(dtype=np.int64)
    t_ts = trigger_df["timestamp"].to_numpy(dtype=np.int64)
    # find insertion index to keep a_ts sorted; subtract 1 to get last index <= t_ts
    idx = np.searchsorted(a_ts, t_ts, side='right') - 1
    out = np.full_like(t_ts, np.nan, dtype=float)

    # mask positions where idx refers to a valid anchor index and also within the series length
    mask = (idx >= 0) & (idx < series.shape[0])
    if not np.any(mask):
        return out

    # For those positions, ensure the chosen anchor timestamp is actually <= the trigger timestamp
    a_idx = idx[mask]
    valid_time = a_ts[a_idx] <= t_ts[mask]

    if not np.any(valid_time):
        return out

    # assign series values for final valid positions
    valid_positions = np.nonzero(mask)[0][valid_time]
    out[valid_positions] = series[a_idx[valid_time]]
    return out

def _load_python_impl(dotted: str) -> Callable:
    # dotted = "module:function"
    if ":" not in dotted:
        raise ValueError(f"Invalid python_impl '{dotted}'. Use 'module:function'")
    mod, fn = dotted.split(":", 1)
    m = importlib.import_module(mod)
    f = getattr(m, fn)
    return f

def _compute_indicator_series_and_threshold(reg: IndicatorRegistry, spec: StrategySpec, filt: IndicatorFilter,
                                            trigger_df: pd.DataFrame, anchor_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ind = reg.get(filt.indicator_id)
    impl = _load_python_impl(ind.python_impl)

    if filt.timeframe == "anchor":
        df = anchor_df
    else:
        df = trigger_df

    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)

    # For MVP: use indicator spec defaults (or midpoints) to keep parity with Pine snippet parameters.
    kwargs = {}
    for p, cfg in (ind.parameters or {}).items():
        if "default" in cfg and cfg["default"] is not None:
            try:
                kwargs[p] = float(cfg["default"]) if cfg.get("type") == "float" else int(float(cfg["default"]))
            except Exception:
                pass
        else:
            lo = cfg.get("min")
            hi = cfg.get("max")
            if lo is not None and hi is not None:
                try:
                    kwargs[p] = float(lo + hi) / 2.0
                except Exception:
                    pass

    outputs = impl(close=close, high=high, low=low, volume=vol, **kwargs)
    if filt.output not in outputs:
        raise KeyError(
            f"Indicator '{ind.indicator_id}' did not produce output '{filt.output}'. outputs={list(outputs.keys())}"
        )
    series = np.asarray(outputs[filt.output], dtype=float)

    # Compute threshold series on the same timeframe as the series (critical for parity with Pine).
    if getattr(filt, "mode", "quantile") == "absolute":
        thr = np.full_like(series, float(getattr(filt, "threshold", 0.0)), dtype=float)
    else:
        q = float(getattr(filt, "quantile", 0.5))
        q = max(0.0, min(1.0, q))
        lookback = int(getattr(filt, "lookback", 500))
        lookback = max(1, lookback)
        thr = percentile_linear_interpolation(series, lookback, q * 100.0)

    # If computed on anchor, align both series and thr to trigger.
    if filt.timeframe == "anchor":
        series = _align_anchor_to_trigger(anchor_df, trigger_df, series)
        thr = _align_anchor_to_trigger(anchor_df, trigger_df, thr)

    return series, thr

# Optional numba JIT path for inner loop
try:
    import numba
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def backtest_mtf_close_to_close(
    trigger_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    symbol: str,
    spec: StrategySpec,
    commission_bps: float,
    slippage_ticks: int = 0,
    tick_size: float | None = None,
    fill_mode: str = "same_bar_close",
    notional: float = 100.0,
    indicator_registry: IndicatorRegistry | None = None,
    use_numba: bool = False,
    precomputed_indicators: Dict | None = None,
) -> BacktestResult:
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    for df, nm in [(trigger_df, "trigger"), (anchor_df, "anchor")]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {nm}: {missing}")

    t_ts = trigger_df["timestamp"].to_numpy(np.int64)
    t_close = trigger_df["close"].to_numpy(float)
    t_high = trigger_df["high"].to_numpy(float)
    t_low = trigger_df["low"].to_numpy(float)

    a_close = anchor_df["close"].to_numpy(float)

    # compute trend on anchor and align
    # Try to use precomputed series when available to avoid repeated computation for many genomes
    key_ema_f = (symbol, spec.tf_anchor, 'ema', int(spec.ema_fast))
    key_ema_s = (symbol, spec.tf_anchor, 'ema', int(spec.ema_slow))
    if precomputed_indicators and key_ema_f in precomputed_indicators:
        a_ema_f = precomputed_indicators[key_ema_f]
    else:
        a_ema_f = ema(a_close, int(spec.ema_fast))
        if precomputed_indicators is not None:
            precomputed_indicators[key_ema_f] = a_ema_f
    if precomputed_indicators and key_ema_s in precomputed_indicators:
        a_ema_s = precomputed_indicators[key_ema_s]
    else:
        a_ema_s = ema(a_close, int(spec.ema_slow))
        if precomputed_indicators is not None:
            precomputed_indicators[key_ema_s] = a_ema_s
    ema_f = _align_anchor_to_trigger(anchor_df, trigger_df, a_ema_f)
    ema_s = _align_anchor_to_trigger(anchor_df, trigger_df, a_ema_s)

    # compute trigger indicators (use precomputed when available)
    key_rsi = (symbol, spec.tf_trigger, 'rsi', int(spec.rsi_len))
    key_atr = (symbol, spec.tf_trigger, 'atr', int(spec.atr_len))
    key_z = (symbol, spec.tf_trigger, 'z', int(spec.z_len))
    if precomputed_indicators and key_rsi in precomputed_indicators:
        r = precomputed_indicators[key_rsi]
    else:
        r = rsi(t_close, int(spec.rsi_len))
        if precomputed_indicators is not None:
            precomputed_indicators[key_rsi] = r
    if precomputed_indicators and key_atr in precomputed_indicators:
        a = precomputed_indicators[key_atr]
    else:
        a = atr(t_high, t_low, t_close, int(spec.atr_len))
        if precomputed_indicators is not None:
            precomputed_indicators[key_atr] = a
    if int(spec.z_len) > 0:
        if precomputed_indicators and key_z in precomputed_indicators:
            z = precomputed_indicators[key_z]
        else:
            z = zscore(t_close, int(spec.z_len))
            if precomputed_indicators is not None:
                precomputed_indicators[key_z] = z
    else:
        z = np.full_like(t_close, np.nan, dtype=float)

    fee_rate = bps_to_rate(commission_bps)
    slippage_ticks = int(slippage_ticks or 0)

    trades: List[Trade] = []
    pos = 0
    entry_i = -1
    entry_price = 0.0
    entry_atr = 0.0

    def next_i(i: int) -> int:
        return i if fill_mode == "same_bar_close" else i + 1

    # precompute extra filter series
    extra_series = []
    if indicator_registry and spec.extra_filters:
        for f in spec.extra_filters:
            s, thr = _compute_indicator_series_and_threshold(indicator_registry, spec, f, trigger_df, anchor_df)
            extra_series.append((f, s, thr))

    # If use_numba is requested and available, use a numba-jitted core to compute trade index pairs (entry_i, exit_i, pos, entry_atr)
    if use_numba and _NUMBA_AVAILABLE:
        # Prepare extra filters arrays as 2D arrays for numba: (n_filters, n)
        n = len(trigger_df)
        nf = len(extra_series)
        ex_vals = np.full((nf, n), np.nan, dtype=float)
        ex_thr = np.full((nf, n), np.nan, dtype=float)
        ex_cmp = np.zeros(nf, dtype=np.int8)  # 0 => '>=', 1 => '<='
        for k, (f, s, thr) in enumerate(extra_series):
            ex_vals[k, :] = s
            ex_thr[k, :] = thr
            ex_cmp[k] = 0 if f.comparator == ">=" else 1

        fill_flag = 0 if fill_mode == "same_bar_close" else 1

        # call numba core
        try:
            core_res = _numba_backtest_core(
                t_close.astype(float),
                ema_f.astype(float),
                ema_s.astype(float),
                r.astype(float),
                a.astype(float),
                z.astype(float),
                n,
                int(spec.z_len), float(spec.z_entry),
                float(spec.rsi_long), float(spec.rsi_short),
                float(spec.sl_atr), float(spec.tp_atr),
                fill_flag,
                ex_vals, ex_thr, ex_cmp,
            )
            # core_res is (count, entry_idx_array, exit_idx_array, pos_array, entry_atr_array)
            count, e_arr, x_arr, p_arr, ea_arr = core_res
            for t_i in range(count):
                entry_i = int(e_arr[t_i])
                j = int(x_arr[t_i])
                pos = int(p_arr[t_i])
                entry_atr = float(ea_arr[t_i])
                # compute prices/fees same as Python version
                if j >= n:
                    break
                exit_side = 'sell' if pos == 1 else 'buy'
                entry_side = 'buy' if pos == 1 else 'sell'
                entry_price = apply_slippage_ticks(t_close[entry_i], tick_size, slippage_ticks, entry_side)
                entry_price = round_to_tick(entry_price, tick_size, mode='nearest')
                exit_price = apply_slippage_ticks(t_close[j], tick_size, slippage_ticks, exit_side)
                exit_price = round_to_tick(exit_price, tick_size, mode='nearest')
                qty = safe_div(notional, entry_price)
                pnl = (exit_price - entry_price) * qty * pos
                fee = (entry_price * qty + exit_price * qty) * fee_rate
                pnl_after = pnl - fee
                r_unit = float(spec.sl_atr) * entry_atr * qty
                pnl_r = pnl_after / r_unit if r_unit != 0 else 0.0
                trades.append(Trade(symbol, spec.tf_trigger, int(t_ts[entry_i]), int(t_ts[j]),
                                    "long" if pos==1 else "short",
                                    float(entry_price), float(exit_price), float(pnl_after), float(pnl_r), float(fee)))
        except Exception:
            # fallback to Python loop if numba core fails for any reason
            pass
    else:
        for i in range(len(trigger_df) - 1):
            if np.isnan(ema_f[i]) or np.isnan(ema_s[i]) or np.isnan(r[i]) or np.isnan(a[i]):
                continue

            trend_up = ema_f[i] > ema_s[i]
            trend_dn = ema_f[i] < ema_s[i]

            z_ok_long = True
            z_ok_short = True
            if int(spec.z_len) > 0 and np.isfinite(z[i]):
                z_ok_long = z[i] >= float(spec.z_entry)
                z_ok_short = z[i] <= -float(spec.z_entry)

            # extra filters: all must pass
            extra_ok_long = True
            extra_ok_short = True
            for f, s, thr in extra_series:
                v = s[i]
                t = thr[i]
                if np.isnan(v) or np.isnan(t):
                    continue
                if f.comparator == ">=":
                    ok = v >= float(t)
                else:
                    ok = v <= float(t)
                # apply to both sides for MVP; can be extended to side-specific filters
                extra_ok_long = extra_ok_long and ok
                extra_ok_short = extra_ok_short and ok

            long_signal = trend_up and (r[i] >= float(spec.rsi_long)) and z_ok_long and extra_ok_long
            short_signal = trend_dn and (r[i] <= float(spec.rsi_short)) and z_ok_short and extra_ok_short

            # exits (close-only)
            if pos != 0:
                sl = entry_price - pos * float(spec.sl_atr) * entry_atr
                tp = entry_price + pos * float(spec.tp_atr) * entry_atr
                exit_now = False
                if pos == 1:
                    if t_close[i] <= sl or t_close[i] >= tp:
                        exit_now = True
                else:
                    if t_close[i] >= sl or t_close[i] <= tp:
                        exit_now = True

                if exit_now:
                    j = next_i(i)
                    if j >= len(trigger_df):
                        break
                    exit_side = 'sell' if pos == 1 else 'buy'
                    exit_price = apply_slippage_ticks(t_close[j], tick_size, slippage_ticks, exit_side)
                    exit_price = round_to_tick(exit_price, tick_size, mode='nearest')
                    exit_price = float(exit_price)
                    exit_price = float(exit_price)
                    qty = safe_div(notional, entry_price)
                    pnl = (exit_price - entry_price) * qty * pos
                    fee = (entry_price * qty + exit_price * qty) * fee_rate
                    pnl_after = pnl - fee
                    r_unit = float(spec.sl_atr) * entry_atr * qty
                    pnl_r = pnl_after / r_unit if r_unit != 0 else 0.0
                    trades.append(Trade(symbol, spec.tf_trigger, int(t_ts[entry_i]), int(t_ts[j]),
                                        "long" if pos==1 else "short",
                                        float(entry_price), float(exit_price), float(pnl_after), float(pnl_r), float(fee)))
                    pos = 0
                    entry_i = -1
                    continue

            # entries (single position)
            if pos == 0:
                if long_signal or short_signal:
                    j = next_i(i)
                    if j >= len(trigger_df):
                        break
                    pos = 1 if long_signal else -1
                    entry_i = j
                    entry_side = 'buy' if pos == 1 else 'sell'
                    entry_price = apply_slippage_ticks(t_close[j], tick_size, slippage_ticks, entry_side)
                    entry_price = round_to_tick(entry_price, tick_size, mode='nearest')
                    entry_price = float(entry_price)
                    entry_atr = float(a[j]) if np.isfinite(a[j]) else float(a[i])

    metrics = _compute_metrics(trades)
    equity_curve = np.cumsum(np.array([t.pnl for t in trades], dtype=float)) if trades else np.array([], dtype=float)
    return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)


# numba core implementation (placed after function to keep it out of hot-path when numba not available)
if _NUMBA_AVAILABLE:
    @njit
    def _numba_backtest_core(t_close, ema_f, ema_s, r, a, z, n,
                             z_len, z_entry, rsi_long, rsi_short, sl_atr, tp_atr,
                             fill_flag, ex_vals, ex_thr, ex_cmp):
        # preallocate arrays sized to n/2 trades worst-case
        max_trades = n // 2 + 2
        e_arr = np.empty(max_trades, dtype=np.int64)
        x_arr = np.empty(max_trades, dtype=np.int64)
        p_arr = np.empty(max_trades, dtype=np.int64)
        ea_arr = np.empty(max_trades, dtype=np.float64)

        pos = 0
        entry_i = -1
        entry_price = 0.0
        entry_atr = 0.0
        tcount = 0

        def next_i_nb(i):
            return i if fill_flag == 0 else i + 1

        nf = ex_vals.shape[0] if ex_vals.ndim == 2 else 0

        for i in range(n - 1):
            # basic nan checks
            if ema_f[i] != ema_f[i] or ema_s[i] != ema_s[i] or r[i] != r[i] or a[i] != a[i]:
                continue

            trend_up = ema_f[i] > ema_s[i]
            trend_dn = ema_f[i] < ema_s[i]

            z_ok_long = True
            z_ok_short = True
            if z_len > 0 and z[i] == z[i]:
                z_ok_long = z[i] >= z_entry
                z_ok_short = z[i] <= -z_entry

            extra_ok_long = True
            extra_ok_short = True
            for k in range(nf):
                v = ex_vals[k, i]
                t = ex_thr[k, i]
                if v != v or t != t:
                    continue
                if ex_cmp[k] == 0:
                    ok = v >= t
                else:
                    ok = v <= t
                extra_ok_long = extra_ok_long and ok
                extra_ok_short = extra_ok_short and ok

            long_signal = trend_up and (r[i] >= rsi_long) and z_ok_long and extra_ok_long
            short_signal = trend_dn and (r[i] <= rsi_short) and z_ok_short and extra_ok_short

            # exits
            if pos != 0:
                sl = entry_price - pos * sl_atr * entry_atr
                tp = entry_price + pos * tp_atr * entry_atr
                exit_now = False
                if pos == 1:
                    if t_close[i] <= sl or t_close[i] >= tp:
                        exit_now = True
                else:
                    if t_close[i] >= sl or t_close[i] <= tp:
                        exit_now = True
                if exit_now:
                    j = next_i_nb(i)
                    if j >= n:
                        break
                    e_arr[tcount] = entry_i
                    x_arr[tcount] = j
                    p_arr[tcount] = pos
                    ea_arr[tcount] = entry_atr
                    tcount += 1
                    pos = 0
                    entry_i = -1
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue

            # entries
            if pos == 0:
                if long_signal or short_signal:
                    j = next_i_nb(i)
                    if j >= n:
                        break
                    pos = 1 if long_signal else -1
                    entry_i = j
                    entry_price = t_close[j]
                    entry_atr = a[j] if a[j] == a[j] else a[i]
        return tcount, e_arr, x_arr, p_arr, ea_arr