from __future__ import annotations
import typer
from rich import print
from pathlib import Path
from ga_trader.config import Config
from ga_trader.utils import save_json, ensure_dir
from ga_trader.providers.binance import fetch_binance_usdt_perp_symbols, fetch_top_volume_usdt_perp_symbols

universe_app = typer.Typer(add_completion=False, help="Universe commands")

@universe_app.command("build")
def build_universe(
    config: str = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    out: str = typer.Option("runs/universe.json", "--out", help="Output universe json"),
):
    """Build trading universe.

    Default behavior (no manual symbols):
      - Select top-N Binance USDT-M PERP symbols by 24h quote volume (Binance /fapi/v1/ticker/24hr).

    Output format:
      - symbols: list[dict] each with symbol, baseAsset, quoteAsset, tick_size, step_size, quote_volume_24h (if available)
    """
    cfg = Config.load(Path(config))
    ucfg = cfg.raw.get("universe", {})

    size = int(ucfg.get("size", 80))
    manual = ucfg.get("manual_symbols")

    bcfg = cfg.raw.get("binance", {})
    base_url = str(bcfg.get("base_url", "https://fapi.binance.com"))

    picked = []
    if manual:
        # Manual mode: enrich user-provided symbols with exchange info metadata.
        symset = {str(s).upper() for s in manual}
        meta = fetch_binance_usdt_perp_symbols(base_url=base_url)
        meta_by = {m["symbol"]: m for m in meta}
        for sym in symset:
            if sym in meta_by:
                picked.append(meta_by[sym])
        picked = picked[:size]
    else:
        picked = fetch_top_volume_usdt_perp_symbols(base_url=base_url, topn=size)

    out_path = Path(out)
    ensure_dir(out_path.parent)
    save_json(out_path, {
        "created_at": cfg.raw.get("created_at"),
        "size": size,
        "symbols": picked,
        "notes": {
            "universe_provider": "binance_futures_24h_volume" if not manual else "manual",
            "binance_base_url": base_url,
        }
    })
    print(f"[green]Universe saved:[/green] {out_path} (n={len(picked)})")
