from __future__ import annotations
import typer
from rich import print
from pathlib import Path
import pandas as pd
from ga_trader.parity.tv_csv import load_tv_trades_csv
from ga_trader.utils import load_json, save_json

validate_app = typer.Typer(add_completion=False, help="Validation / parity")

def _match_trades(py_trades, tv_df: pd.DataFrame, ts_tol_ms: int = 60_000):
    '''
    Match trades by (side, entry_time ~, exit_time ~). Returns matched, missing, extra.
    '''
    tv = tv_df.copy()
    tv = tv.fillna({"entry_time_ms": -1, "exit_time_ms": -1})
    used = set()
    matched = 0
    mismatches = []

    for i, t in enumerate(py_trades):
        side = t.get("side")
        et = int(t.get("entry_ts", -1))
        xt = int(t.get("exit_ts", -1))
        best = None
        best_j = None
        for j, row in tv.iterrows():
            if j in used:
                continue
            # direction heuristic
            d = str(row.get("direction","")).lower()
            if side == "long" and d and "long" not in d and "buy" not in d:
                continue
            if side == "short" and d and "short" not in d and "sell" not in d:
                continue
            dt_e = abs(int(row.get("entry_time_ms",-1)) - et) if row.get("entry_time_ms",-1) != -1 else 10**12
            dt_x = abs(int(row.get("exit_time_ms",-1)) - xt) if row.get("exit_time_ms",-1) != -1 else 10**12
            if dt_e <= ts_tol_ms and dt_x <= ts_tol_ms:
                score = dt_e + dt_x
                if best is None or score < best:
                    best = score
                    best_j = j
        if best_j is not None:
            used.add(best_j)
            matched += 1
        else:
            mismatches.append({"python_trade_index": i, "reason": "no_match", "trade": t})
    extra = int(len(tv) - len(used))
    return matched, mismatches, extra

@validate_app.command("pine")
def validate_pine(
    python_trades_json: str = typer.Option(..., "--python_trades_json", help="JSON with python trades exported by your run"),
    tv_trades_csv: str = typer.Option(..., "--tv_trades_csv", help="TradingView exported trades CSV (List of Trades)"),
    out: str = typer.Option("parity_report.json", "--out", help="Output report json"),
    ts_tol_ms: int = typer.Option(60000, "--ts_tol_ms", help="Timestamp tolerance for matching trades"),
):
    py = load_json(Path(python_trades_json))
    py_trades = py.get("trades", [])
    tv = load_tv_trades_csv(Path(tv_trades_csv))

    py_pnl = sum(float(t.get("pnl", 0.0)) for t in py_trades)
    tv_pnl = float(tv["pnl"].sum()) if "pnl" in tv.columns else float("nan")

    matched, mismatches, extra = _match_trades(py_trades, tv, ts_tol_ms=ts_tol_ms)

    report = {
        "python_trades": int(len(py_trades)),
        "tv_trades": int(len(tv)),
        "matched_trades": int(matched),
        "unmatched_python_trades": int(len(mismatches)),
        "extra_tv_trades": int(extra),
        "python_net_pnl": float(py_pnl),
        "tv_net_pnl": float(tv_pnl),
        "mismatches_sample": mismatches[:10],
        "notes": "For best parity: TV strategy process_orders_on_close=true, same fee/slippage, identical dataset.",
    }
    save_json(Path(out), report)
    print(f"[green]Saved parity report[/green]: {out}")
