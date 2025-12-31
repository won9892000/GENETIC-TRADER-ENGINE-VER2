from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import re
import yaml


def _load_indicator_specs(root: Path) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    specs_dir = root / "specs"
    if specs_dir.exists():
        for p in specs_dir.glob("*.yaml"):
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            if raw and "indicator_id" in raw:
                specs[str(raw["indicator_id"])] = raw
    return specs


def render_pine_strategy(
    spec: Dict[str, Any],
    name: str = "GA_Strategy",
    indicators_root: str = "indicators",
    commission_bps: float = 0.0,
    slippage_ticks: int = 0,
) -> str:
    """Render a close-to-close Pine v5 strategy.

    Parity design:
    - Signal computed on bar close
    - Orders executed on bar close (process_orders_on_close=true)
    - SL/TP evaluated on close only (manual exit on close), matching Python engine
    - Extra filters: absolute threshold or rolling percentile (ta.percentile_linear_interpolation)
    """

    tf_anchor = str(spec.get("tf_anchor", "15m"))
    tf_trigger = str(spec.get("tf_trigger", "5m"))
    commission_pct = float(commission_bps) / 100.0
    slippage_ticks = int(slippage_ticks or 0)

    ef = int(spec.get("ema_fast", 20))
    es = int(spec.get("ema_slow", 50))
    rl = int(spec.get("rsi_len", 14))
    rlong = float(spec.get("rsi_long", 55.0))
    rshort = float(spec.get("rsi_short", 45.0))
    al = int(spec.get("atr_len", 14))
    sl = float(spec.get("sl_atr", 1.5))
    tp = float(spec.get("tp_atr", 1.5))
    zl = int(spec.get("z_len", 0) or 0)
    zentry = float(spec.get("z_entry", 0.0) or 0.0)

    extra_filters = spec.get("extra_filters", []) or []
    ind_specs = _load_indicator_specs(Path(indicators_root))

    anchor_block = f"""anchorTF = input.timeframe("{tf_anchor}", "Anchor TF")
triggerTF = input.timeframe("{tf_trigger}", "Trigger TF (chart TF should match)")

emaFastLen = input.int({ef}, "EMA Fast", minval=1)
emaSlowLen = input.int({es}, "EMA Slow", minval=2)

emaF = request.security(syminfo.tickerid, anchorTF, ta.ema(close, emaFastLen))
emaS = request.security(syminfo.tickerid, anchorTF, ta.ema(close, emaSlowLen))
trendUp = emaF > emaS
trendDn = emaF < emaS
"""

    trigger_block = f"""rsiLen = input.int({rl}, "RSI Length", minval=1)
rsiLong = input.float({rlong}, "RSI Long Trigger", minval=0, maxval=100)
rsiShort = input.float({rshort}, "RSI Short Trigger", minval=0, maxval=100)
atrLen = input.int({al}, "ATR Length", minval=1)
slATR = input.float({sl}, "SL ATR", minval=0.1)
tpATR = input.float({tp}, "TP ATR", minval=0.1)

r = ta.rsi(close, rsiLen)
a = ta.atr(atrLen)
"""

    z_block = ""
    if zl > 0:
        z_block = f"""// Z-score filter (optional)
zLen = input.int({zl}, "Z Length", minval=1)
zEntry = input.float({zentry}, "Z Entry Threshold", step=0.1)
z = (close - ta.sma(close, zLen)) / ta.stdev(close, zLen)
zLongOK = z <= -zEntry
zShortOK = z >= zEntry
"""
    else:
        z_block = "zLongOK = true\nzShortOK = true\n"

    extra_blocks: List[str] = []
    extra_conditions_long: List[str] = []
    extra_conditions_short: List[str] = []

    for idx, f in enumerate(extra_filters):
        iid = str(f.get("indicator_id"))
        out = str(f.get("output"))
        comp = str(f.get("comparator", ">=")).strip()
        tf = str(f.get("timeframe", "trigger")).strip()
        mode = str(f.get("mode", "quantile")).strip()
        thr_abs = float(f.get("threshold", 0.0))
        q = float(f.get("quantile", 0.5))
        lb = int(f.get("lookback", 500))
        q = max(0.0, min(1.0, q))
        lb = max(1, lb)
        pct = q * 100.0

        raw = ind_specs.get(iid)
        if not raw:
            continue
        pine_snippet = str(raw.get("pine_snippet", "")).rstrip()
        params = raw.get("parameters", {}) or {}

        # parameter inputs
        decls: List[str] = []
        for p, cfg in params.items():
            typ = cfg.get("type", "int")
            lo = cfg.get("min")
            hi = cfg.get("max")
            default = cfg.get("default")
            if default is None and lo is not None and hi is not None:
                default = (float(lo) + float(hi)) / 2.0
            if typ == "int":
                decls.append(f'{p}_{idx} = input.int({int(float(default or 14))}, "{iid}:{p}", minval={int(lo) if lo is not None else 1})')
            elif typ == "float":
                decls.append(f'{p}_{idx} = input.float({float(default or 14.0)}, "{iid}:{p}", minval={float(lo) if lo is not None else 0.0})')
            elif typ == "bool":
                decls.append(f'{p}_{idx} = input.bool({str(default).lower() if default is not None else "false"}, "{iid}:{p}")')

        # rewrite snippet param names to unique per filter
        rewritten = pine_snippet
        for p in params.keys():
            rewritten = re.sub(rf"\b{re.escape(p)}\b", f"{p}_{idx}", rewritten)

        val_var = f"val_{idx}"
        thr_var = f"thr_{idx}"

        if tf == "anchor":
            # define local function for the indicator output
            fn_name = f"f_{idx}"
            fn_lines = [f"{fn_name}() =>"]
            for ln in rewritten.splitlines():
                fn_lines.append("    " + ln)
            fn_lines.append("    " + f"return {out}")
            fn_block = "\n".join(fn_lines)

            extra_blocks.append("\n".join(decls))
            extra_blocks.append(fn_block)
            extra_blocks.append(f"{val_var} = request.security(syminfo.tickerid, anchorTF, {fn_name}())")
            if mode == "absolute":
                extra_blocks.append(f"{thr_var} = {thr_abs}")
            else:
                extra_blocks.append(f"{thr_var} = request.security(syminfo.tickerid, anchorTF, ta.percentile_linear_interpolation({fn_name}(), {lb}, {pct}))")
        else:
            extra_blocks.append("\n".join(decls))
            extra_blocks.append(rewritten)
            extra_blocks.append(f"{val_var} = {out}")
            if mode == "absolute":
                extra_blocks.append(f"{thr_var} = {thr_abs}")
            else:
                extra_blocks.append(f"{thr_var} = ta.percentile_linear_interpolation({val_var}, {lb}, {pct})")

        extra_blocks.append(f"// Extra filter {idx}: {iid}.{out} ({tf}, {mode})")

        extra_conditions_long.append(f"({val_var} {comp} {thr_var})")
        extra_conditions_short.append(f"({val_var} {comp} {thr_var})")

    extra_cond_long = " and ".join(extra_conditions_long) if extra_conditions_long else "true"
    extra_cond_short = " and ".join(extra_conditions_short) if extra_conditions_short else "true"
    extra_block = "\n".join([b for b in extra_blocks if b.strip()])

    pine = f"""//@version=5
strategy("{name}", overlay=true,
     initial_capital=10000,
     commission_type=strategy.commission.percent,
     commission_value={commission_pct},
     slippage={slippage_ticks},
     process_orders_on_close=true,
     calc_on_every_tick=false,
     pyramiding=0)

{anchor_block}
{trigger_block}
{z_block}

// Extra indicator filters
{extra_block}

longSignal = trendUp and (r >= rsiLong) and zLongOK and ({extra_cond_long})
shortSignal = trendDn and (r <= rsiShort) and zShortOK and ({extra_cond_short})

// Close-only position management for parity with Python engine
var float entryPrice = na
var float entryATR = na

if strategy.position_size == 0
    entryPrice := na
    entryATR := na
    if longSignal
        strategy.entry("L", strategy.long)
    if shortSignal
        strategy.entry("S", strategy.short)

if strategy.position_size != 0 and na(entryPrice)
    entryPrice := strategy.position_avg_price
    entryATR := a

if strategy.position_size > 0
    slPrice = entryPrice - slATR * entryATR
    tpPrice = entryPrice + tpATR * entryATR
    if close <= slPrice or close >= tpPrice
        strategy.close("L")

if strategy.position_size < 0
    slPrice = entryPrice + slATR * entryATR
    tpPrice = entryPrice - tpATR * entryATR
    if close >= slPrice or close <= tpPrice
        strategy.close("S")
"""
    return pine
