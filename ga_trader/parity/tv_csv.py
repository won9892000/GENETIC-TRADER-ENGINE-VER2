from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
from dateutil import parser as dtparser

def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read TradingView CSV with delimiter/encoding sniffing.

    TradingView exports can use comma/semicolon/tab delimiters depending on locale.
    """
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    seps = [None, ",", ";", "\t"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc)
                return pd.read_csv(path, sep=sep, encoding=enc)
            except Exception as e:
                last_err = e
                continue
    # fall back without hints
    if last_err:
        raise last_err
    return pd.read_csv(path)

def _norm(s: str) -> str:
    # keep non-latin chars (e.g., Korean) but normalize whitespace/punctuation
    return s.strip().lower().replace(" ", "_").replace("%", "pct").replace("/", "_")

def _to_num(x):
    if pd.isna(x):
        return float("nan")
    s = str(x).strip()

    # Handle European-style decimals: "1,23" -> "1.23" when dot absent.
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    # Remove currency symbols and thousands separators (commas) when dot present.
    if "." in s:
        s = s.replace(",", "")

    s = re.sub(r"[^0-9\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return float("nan")

def _to_ts_ms(x):
    if pd.isna(x):
        return None
    try:
        dt = dtparser.parse(str(x))
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

def _pick_col(cols: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None

def load_tv_trades_csv(path: Path) -> pd.DataFrame:
    """Robust-ish TradingView 'List of Trades' CSV loader.

    Normalizes common columns into:
      entry_time_ms, exit_time_ms, direction, qty, entry_price, exit_price, pnl

    Notes:
    - TradingView CSV column names vary by locale and strategy settings.
    - This loader uses regex heuristics rather than hard-coded headers.
    """
    df = _read_csv_robust(path)
    df = df.rename(columns={c: _norm(c) for c in df.columns})
    cols = list(df.columns)

    # Column heuristics (English + common KR labels)
    entry_time_col = _pick_col(cols, [
        r"^entry(_|\b).*time", r"entry_time", r"진입.*시간", r"진입시간"
    ])
    exit_time_col = _pick_col(cols, [
        r"^exit(_|\b).*time", r"exit_time", r"청산.*시간", r"청산시간", r"종료.*시간"
    ])
    pnl_col = _pick_col(cols, [
        r"(profit|pnl|net_profit|gross_profit)", r"손익", r"수익", r"이익", r"손실"
    ])
    dir_col = _pick_col(cols, [
        r"^(direction|side|position|trade_direction)$", r"방향", r"포지션", r"구분"
    ])
    qty_col = _pick_col(cols, [
        r"^(qty|quantity|contracts|size)$", r"수량", r"계약"
    ])
    entry_price_col = _pick_col(cols, [
        r"^(entry_price|avg_entry_price|entry)$", r"진입가", r"진입_가격"
    ])
    exit_price_col = _pick_col(cols, [
        r"^(exit_price|avg_exit_price|exit)$", r"청산가", r"청산_가격"
    ])

    if entry_time_col:
        df["entry_time_ms"] = df[entry_time_col].apply(_to_ts_ms)
    else:
        df["entry_time_ms"] = None

    if exit_time_col:
        df["exit_time_ms"] = df[exit_time_col].apply(_to_ts_ms)
    else:
        df["exit_time_ms"] = None

    if pnl_col:
        df["pnl"] = df[pnl_col].apply(_to_num)
    else:
        df["pnl"] = float("nan")

    if dir_col:
        d = df[dir_col].astype(str).str.lower()
        # Normalize to 'long'/'short' if possible
        d = d.replace({
            "buy": "long", "sell": "short",
            "롱": "long", "숏": "short",
            "long": "long", "short": "short",
        })
        df["direction"] = d
    else:
        df["direction"] = None

    if qty_col:
        df["qty"] = df[qty_col].apply(_to_num)
    else:
        df["qty"] = float("nan")

    if entry_price_col:
        df["entry_price"] = df[entry_price_col].apply(_to_num)
    else:
        df["entry_price"] = float("nan")

    if exit_price_col:
        df["exit_price"] = df[exit_price_col].apply(_to_num)
    else:
        df["exit_price"] = float("nan")

    return df
