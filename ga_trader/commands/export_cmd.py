from __future__ import annotations
import typer
from rich import print
from pathlib import Path
from ga_trader.utils import load_json, ensure_dir
from ga_trader.config import Config
from ga_trader.export.pine import render_pine_strategy

export_app = typer.Typer(add_completion=False, help="Export commands")

@export_app.command("pine")
def export_pine(
    config: str = typer.Option(..., "--config", "-c", help="config.yaml"),
    run_dir: str = typer.Option(..., "--run_dir", help="Run directory containing topk.json"),
    topk: int = typer.Option(20, "--topk", help="How many strategies to export"),
    out: str = typer.Option("pine_out", "--out", help="Output directory for pine scripts"),
    indicators_root: str = typer.Option("indicators", "--indicators_root", help="Indicator root folder"),
):
    cfg = Config.load(config)
    trading = cfg.raw.get('trading', {})
    commission_bps = float(trading.get('commission_bps', 0.0))
    exec_cfg = cfg.raw.get('execution', {})
    slippage_ticks = int(exec_cfg.get('slippage_ticks', 0))

    run_path = Path(run_dir)
    top = load_json(run_path / "topk.json")
    out_dir = Path(out)
    ensure_dir(out_dir)

    for i, item in enumerate(top[:topk]):
        spec = item["spec"]
        pine = render_pine_strategy(spec, name=f"GA_Strategy_{i:03d}", indicators_root=indicators_root, commission_bps=commission_bps, slippage_ticks=slippage_ticks)
        p = out_dir / f"GA_Strategy_{i:03d}.pine"
        p.write_text(pine, encoding="utf-8")
    print(f"[green]Exported[/green] {min(topk, len(top))} pine scripts to {out_dir}")