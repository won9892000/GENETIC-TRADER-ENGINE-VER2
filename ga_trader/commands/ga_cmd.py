from __future__ import annotations
import typer
from rich import print
from pathlib import Path
import pandas as pd

from ga_trader.config import Config
from ga_trader.utils import load_json, ensure_dir, save_json
from ga_trader.ga.runner import run_ga, EvalContext
from ga_trader.indicators.registry import IndicatorRegistry

ga_app = typer.Typer(add_completion=False, help="GA commands")

@ga_app.command("run")
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    universe: str = typer.Option(..., "--universe", "-u", help="Universe json"),
    indicators_root: str = typer.Option("indicators", "--indicators_root", help="Indicator root (specs/)"),
    workers: int | None = typer.Option(None, "--workers", "-w", help="Number of worker threads to use (defaults to CPU-1)"),
    cache_file: str | None = typer.Option(None, "--cache-file", help="Optional path to persistent cache file for evaluated specs"),
    use_processes: bool = typer.Option(False, "--use-processes", help="Use multiprocessing instead of threading for genome evals"),
    use_numba: bool = typer.Option(False, "--use-numba", help="Enable numba JIT for inner backtest loop if available"),
):
    cfg = Config.load(config)
    u = load_json(Path(universe))
    raw_symbols = u.get("symbols", [])
    symbols: list[str] = []
    tick_size_by_symbol: dict[str, float] = {}
    for it in raw_symbols:
        if isinstance(it, dict):
            sym = str(it.get("symbol"))
            symbols.append(sym)
            if "tick_size" in it and it["tick_size"] is not None:
                tick_size_by_symbol[sym] = float(it["tick_size"])
        else:
            symbols.append(str(it))
    if not symbols:
        raise ValueError("Universe has no symbols")
    
    # Backward compatibility: if universe includes top-level tick_size_by_symbol
    if not tick_size_by_symbol and isinstance(u.get("tick_size_by_symbol"), dict):
        tick_size_by_symbol = {str(k): float(v) for k, v in u["tick_size_by_symbol"].items()}

    dcfg = cfg.raw.get("data", {})
    tfs = list(dcfg.get("timeframes", ["5m"]))
    store_dir = Path(dcfg.get("store_dir", "data"))
    split = dcfg.get("split", {"is": 0.6, "oos": 0.2, "hold": 0.2})

    exec_cfg = cfg.raw.get("execution", {})
    fill_mode = str(exec_cfg.get("fill", "same_bar_close"))
    slippage_ticks = int(exec_cfg.get("slippage_ticks", 0))
    # backward compat
    if "slippage_bps" in exec_cfg and slippage_ticks == 0:
        # approximate: convert bps to ticks later (requires tick_size), keep here as bps for warning
        pass

    risk_cfg = cfg.raw.get("risk", {})
    notional = float(risk_cfg.get("fixed_notional", 100.0))

    # load indicators registry (optional)
    ind_root = Path(indicators_root)
    reg = IndicatorRegistry.from_root(ind_root) if ind_root.exists() else None
    if reg and reg.list():
        print(f"[green]Loaded indicators[/green]: {len(reg.list())} from {ind_root}")
    else:
        reg = None
        print("[yellow]No indicator registry found or empty; GA will run without extra indicator filters.[/yellow]")

    data: dict[str, dict[str, pd.DataFrame]] = {}
    missing = []
    for sym in symbols:
        s = sym["symbol"]
        tmap = {}
        for tf in tfs:
            path = store_dir / f"{s}_{tf}.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            df = pd.read_csv(path)
            need = {"timestamp","open","high","low","close","volume"}
            if not need.issubset(df.columns):
                raise ValueError(f"{path} missing columns: {need - set(df.columns)}")
            tmap[tf] = df
        if tmap:
            data[s] = tmap

    if missing:
        print("[yellow]Missing data files (run data fetch first):[/yellow]")
        for m in missing[:10]:
            print(" -", m)
        raise typer.Exit(code=1)

    gcfg = cfg.raw.get("ga", {})
    fcfg = cfg.raw.get("fitness", {})
    space = cfg.raw.get("strategy_space", {})

    run_root = Path(cfg.raw.get("project", {}).get("run_root", "runs"))
    run_name = str(cfg.raw.get("project", {}).get("run_name", "latest"))
    out_dir = run_root / run_name
    ensure_dir(out_dir)

    save_json(out_dir / "universe_snapshot.json", u)
    (out_dir / "config_snapshot.yaml").write_text(cfg.path.read_text(encoding="utf-8"), encoding="utf-8")

    ctx = EvalContext(
        data=data,
        commission_bps=float(cfg.raw.get("fees", {}).get("commission_bps", 0.0)),
        slippage_ticks=slippage_ticks,
        tick_size_by_symbol=tick_size_by_symbol,
        fill_mode=fill_mode,
        notional=notional,
        split=split,
        timeframes=tfs,
        indicators=reg,
        parity_symbol=cfg.parity_symbol,
        use_numba=use_numba,
    )

    print(f"[green]Running GA[/green] pop={gcfg.get('population')} gens={gcfg.get('generations')} on symbols={len(data)}")
    run_ga(
        space=space,
        ctx=ctx,
        fitness_cfg=fcfg,
        population=int(gcfg.get("population", 160)),
        generations=int(gcfg.get("generations", 60)),
        elite_ratio=float(gcfg.get("elite_ratio", 0.05)),
        tournament_k=int(gcfg.get("tournament_k", 5)),
        crossover_rate=float(gcfg.get("crossover_rate", 0.7)),
        mutation_rate=float(gcfg.get("mutation_rate", 0.18)),
        rng_seed=int(gcfg.get("seed", 42)),
        out_dir=out_dir,
        topk=int(gcfg.get("topk", 20)),
        workers=workers,
        cache_file=cache_file,
    )
    print(f"[green]GA completed[/green]. Run dir: {out_dir}")