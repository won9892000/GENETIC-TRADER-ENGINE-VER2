from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich import print

from ga_trader.config import Config, get
from ga_trader.providers.binance import fetch_klines
from ga_trader.utils import ensure_dir, load_json

data_app = typer.Typer(add_completion=False, help="Data commands")


def _normalize_universe_symbols(u) -> list[str]:
    """Normalize supported universe formats into a list of symbol strings.

    Supported universe formats:
      1) {"symbols": [{"symbol": "BTCUSDT", ...}, ...], ...}
      2) [{"symbol": "BTCUSDT", ...}, ...]
      3) ["BTCUSDT", "ETHUSDT", ...]
    """
    symbols_raw = u.get("symbols", u) if isinstance(u, dict) else u

    if isinstance(symbols_raw, list):
        if len(symbols_raw) == 0:
            return []
        if isinstance(symbols_raw[0], dict):
            return [row["symbol"] for row in symbols_raw if isinstance(row, dict) and "symbol" in row]
        return [str(x) for x in symbols_raw]

    raise ValueError(f"Unsupported universe format: {type(u)} (expected dict or list)")


@data_app.command("fetch")
def fetch_data(
    config: str = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    universe: str = typer.Option(..., "--universe", "-u", help="Universe json produced by universe build"),
):
    """Fetch Binance klines for symbols in the universe and save them as CSV."""
    cfg = Config.load(config)
    u = load_json(Path(universe))
    symbols = _normalize_universe_symbols(u)

    if not symbols:
        raise typer.BadParameter("Universe contains no symbols.")

    # Data settings
    tfs = list(get(cfg.raw, "data.timeframes", ["3m", "5m", "15m"]))
    lookback_days = int(get(cfg.raw, "data.lookback_days", 180))
    store_dir = Path(get(cfg.raw, "data.store_dir", "data"))

    # Binance settings
    base_url = str(get(cfg.raw, "binance.base_url", "https://fapi.binance.com"))
    sleep_sec = float(get(cfg.raw, "binance.rate_limit_sleep_sec", 0.15))

    ensure_dir(store_dir)

    total = 0
    for s in symbols:
        for tf in tfs:
            df = fetch_klines(base_url, s, tf, lookback_days=lookback_days, sleep_sec=sleep_sec)
            out_path = store_dir / f"{s}_{tf}.csv"
            df.to_csv(out_path, index=False)
            total += 1
            print(f"[cyan]Saved[/cyan] {out_path} rows={len(df)}")

    print(f"[green]Done[/green]. Files written: {total}")
