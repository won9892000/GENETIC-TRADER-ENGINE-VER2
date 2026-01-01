from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import random
import numpy as np
import pandas as pd
from pathlib import Path
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ga_trader.ga.genome import Genome, random_genome, crossover, mutate
from ga_trader.backtest.engine import backtest_mtf_close_to_close
from ga_trader.fitness.evaluator import evaluate_fitness, FitnessResult
from ga_trader.utils import ensure_dir, save_json
from ga_trader.indicators.registry import IndicatorRegistry
from ga_trader.strategy.spec import StrategySpec, strategy_spec_from_dict
from ga_trader.ga.nsga2 import nsga2_select

def _trade_to_dict(t) -> dict:
    return {
        "symbol": t.symbol,
        "timeframe": t.timeframe,
        "entry_ts": int(t.entry_ts),
        "exit_ts": int(t.exit_ts),
        "side": t.side,
        "entry_price": float(t.entry_price),
        "exit_price": float(t.exit_price),
        "pnl": float(t.pnl),
        "pnl_r": float(t.pnl_r),
        "fee": float(t.fee),
    }


@dataclass
class EvalContext:
    # data: symbol -> timeframe -> df
    data: Dict[str, Dict[str, pd.DataFrame]]
    commission_bps: float
    slippage_ticks: int
    tick_size_by_symbol: Dict[str, float]
    fill_mode: str
    notional: float
    split: Dict[str, float]
    timeframes: List[str]
    indicators: IndicatorRegistry | None
    parity_symbol: str | None = None
    # optional fast path: use numba JIT for inner backtest loop when available
    use_numba: bool = False

def _split_indices(n: int, split: Dict[str, float]) -> Dict[str, slice]:
    is_ratio = float(split.get("is", 0.6))
    oos_ratio = float(split.get("oos", 0.2))
    i1 = int(n * is_ratio)
    i2 = int(n * (is_ratio + oos_ratio))
    return {
        "is": slice(0, max(0, i1)),
        "oos": slice(max(0, i1), max(0, i2)),
        "hold": slice(max(0, i2), n),
    }

# multiprocessing helpers (module-level so they are picklable)
_proc_ctx = None
_proc_fcfg = None

def _proc_init(c, f):
    global _proc_ctx, _proc_fcfg
    _proc_ctx = c
    _proc_fcfg = f


def _proc_score(g):
    if _proc_ctx is None or _proc_fcfg is None:
        raise RuntimeError("Process worker not initialized")
    f, extra = evaluate_genome(g, _proc_ctx, _proc_fcfg)
    return (g, f.fitness, extra)

def evaluate_genome(g: Genome, ctx: EvalContext, fitness_cfg: Dict[str, Any]) -> Tuple[FitnessResult, Dict[str, Any]]:
    spec = g.to_spec()
    agg = {"is": [], "oos": []}

    for sym, tmap in ctx.data.items():
        if spec.tf_trigger not in tmap or spec.tf_anchor not in tmap:
            continue
        trigger_df = tmap[spec.tf_trigger]
        anchor_df = tmap[spec.tf_anchor]
        n = min(len(trigger_df), len(anchor_df))
        if n < 300:
            continue

        # split based on trigger length
        slices = _split_indices(len(trigger_df), ctx.split)

        for part in ["is", "oos"]:
            tdf = trigger_df.iloc[slices[part]].reset_index(drop=True)
            # choose anchor slice by timestamp overlap
            min_ts = int(tdf["timestamp"].iloc[0])
            max_ts = int(tdf["timestamp"].iloc[-1])
            adf = anchor_df[(anchor_df["timestamp"] >= min_ts) & (anchor_df["timestamp"] <= max_ts)].reset_index(drop=True)
            if len(adf) < 50 or len(tdf) < 200:
                continue

            bt = backtest_mtf_close_to_close(
                trigger_df=tdf,
                anchor_df=adf,
                symbol=sym,
                spec=spec,
                commission_bps=ctx.commission_bps,
                slippage_ticks=ctx.slippage_ticks,
                tick_size=ctx.tick_size_by_symbol.get(sym),
                fill_mode=ctx.fill_mode,
                notional=ctx.notional,
                indicator_registry=ctx.indicators,
                use_numba=bool(ctx.use_numba),
            )
            agg[part].append(bt.metrics)

    def mean_metric(part: str, key: str) -> float:
        vals = [m.get(key, 0.0) for m in agg[part]]
        return float(np.mean(vals)) if vals else 0.0

    metrics = {
        "is_expectancy": mean_metric("is", "expectancy"),
        "is_profit_factor": mean_metric("is", "profit_factor"),
        "is_win_rate": mean_metric("is", "win_rate"),
        "is_trades_per_month": mean_metric("is", "trades_per_month"),
        "is_max_drawdown": mean_metric("is", "max_drawdown"),
        "oos_expectancy": mean_metric("oos", "expectancy"),
        "oos_profit_factor": mean_metric("oos", "profit_factor"),
        "oos_win_rate": mean_metric("oos", "win_rate"),
        "oos_trades_per_month": mean_metric("oos", "trades_per_month"),
        "oos_max_drawdown": mean_metric("oos", "max_drawdown"),
    }
    metrics["is_oos_gap"] = abs(metrics["is_expectancy"] - metrics["oos_expectancy"])

    mode = str(fitness_cfg.get("mode", "weighted_sum"))
    if mode == "targets":
        weights = dict(fitness_cfg.get("targets", {}) or {})
    else:
        weights = dict(fitness_cfg.get("weights", {}) or {})
    fit = evaluate_fitness(metrics, mode=mode,
                           weights=weights,
                           constraints=dict(fitness_cfg.get("constraints", {}) or {}))
    extra = {"spec": spec.to_dict(), "metrics": metrics}
    return fit, extra

def tournament_select(pop: List[Tuple[Genome, float]], k: int, rng: random.Random) -> Genome:
    cand = rng.sample(pop, k=min(k, len(pop)))
    cand.sort(key=lambda x: x[1], reverse=True)
    return cand[0][0]

def run_ga(
    space: Dict[str, Any],
    ctx: EvalContext,
    fitness_cfg: Dict[str, Any],
    population: int,
    generations: int,
    elite_ratio: float,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate: float,
    rng_seed: int,
    out_dir: Path,
    topk: int = 20,
    workers: int | None = None,
    cache_file: str | None = None,
    use_processes: bool = False,
):
    rng = random.Random(int(rng_seed))
    ensure_dir(out_dir)

    reg = ctx.indicators

    pop = [random_genome(space, rng, ctx.timeframes, reg) for _ in range(population)]
    history = []
    import pickle
    scored_cache = {}
    # optional persistent cache
    if 'cache_file' in locals() and locals().get('cache_file'):
        try:
            with open(locals().get('cache_file'), 'rb') as _f:
                scored_cache = pickle.load(_f) or {}
        except Exception:
            scored_cache = {}

    def cache_key(g: Genome):
        d = g.to_spec().to_dict()
        # stable key for dict + extra_filters
        ef = tuple((f["indicator_id"], f["output"], f["comparator"], round(f["threshold"],6), f["timeframe"]) for f in d.get("extra_filters", []))
        return (d["tf_anchor"], d["tf_trigger"], d["ema_fast"], d["ema_slow"], d["rsi_len"], round(d["rsi_long"],4), round(d["rsi_short"],4),
                d["atr_len"], round(d["sl_atr"],4), round(d["tp_atr"],4), d["z_len"], round(d["z_entry"],4), ef)

    cache_lock = threading.Lock()

    def score(g: Genome) -> Tuple[float, Dict[str, Any]]:
        key = cache_key(g)
        with cache_lock:
            if key in scored_cache:
                return scored_cache[key][0], scored_cache[key][1]
        fit, extra = evaluate_genome(g, ctx, fitness_cfg)
        with cache_lock:
            scored_cache[key] = (fit.fitness, {"fitness_details": fit.details, **extra})
            return scored_cache[key][0], scored_cache[key][1]



    pareto_cfg = dict((fitness_cfg.get("pareto") or {}))
    objectives = pareto_cfg.get("objectives") or [
        {"key":"oos_expectancy","direction":"max"},
        {"key":"oos_max_drawdown","direction":"min"},
        {"key":"oos_trades_per_month","direction":"max"},
    ]

    for gen in range(generations):
        scored = []
        # determine worker count: explicit param > env var > cpu_count
        if workers is None:
            try:
                workers = int(os.environ.get("GA_WORKERS", "0")) or max(1, (os.cpu_count() or 1) - 1)
            except Exception:
                workers = max(1, (os.cpu_count() or 1) - 1)
        if workers and workers > 1:
            if use_processes:
                # Use multiprocessing Pool with initializer to avoid pickling ctx per task
                from multiprocessing import Pool
                # Build list of genomes that need evaluation (not in cache)
                to_eval = []
                idx_map = {}
                for idx, g in enumerate(pop):
                    k = cache_key(g)
                    if k not in scored_cache:
                        idx_map[len(to_eval)] = idx
                        to_eval.append(g)
                    else:
                        scored.append((g, scored_cache[k][0], scored_cache[k][1]))
                if to_eval:
                    with Pool(processes=workers, initializer=_proc_init, initargs=(ctx, fitness_cfg)) as p:
                        results = p.map(_proc_score, to_eval)
                        for res in results:
                            g, fval, extra = res
                            k = cache_key(g)
                            with cache_lock:
                                scored_cache[k] = (fval, extra)
                            original_idx = None
                            # find original genome location(s) and append
                            scored.append((g, fval, extra))
            else:
                # ThreadPool is a good fit: numpy releases GIL during heavy computations and avoids large pickling costs.
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    future_to_gen = {ex.submit(score, g): g for g in pop}
                    for fut in as_completed(future_to_gen):
                        g = future_to_gen[fut]
                        f, extra = fut.result()
                        scored.append((g, f, extra))
        else:
            for g in pop:
                f, extra = score(g)
                scored.append((g, f, extra))
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        history.append({
            "generation": gen,
            "best_fitness": float(best[1]),
            "best_spec": best[2]["spec"],
            "best_metrics": best[2]["metrics"],
        })

        elite_n = max(1, int(population * elite_ratio))
        elites = [x[0] for x in scored[:elite_n]]

        mode = str(fitness_cfg.get("mode", "weighted_sum"))
        new_pop = elites[:]

        if mode == "pareto":
            # NSGA-II selection on metrics (use OOS metrics by default)
            score_dicts = []
            for _, _, extra in scored:
                score_dicts.append(extra["metrics"])
            obj_list = [(o["key"], o.get("direction","max")) for o in objectives]
            selected_idx = nsga2_select(score_dicts, obj_list, pop_size=population-elite_n)
            selected = [scored[i][0] for i in selected_idx]
            # breed from selected set
            pool = [(g, 0.0) for g in selected]
            while len(new_pop) < population:
                p1 = tournament_select(pool, tournament_k, rng)
                p2 = tournament_select(pool, tournament_k, rng)
                child = crossover(p1, p2, rng) if rng.random() < crossover_rate else p1
                child = mutate(child, space, rng, mutation_rate, ctx.timeframes, reg)
                new_pop.append(child)
        else:
            pool = [(x[0], x[1]) for x in scored]
            while len(new_pop) < population:
                p1 = tournament_select(pool, tournament_k, rng)
                p2 = tournament_select(pool, tournament_k, rng)
                child = crossover(p1, p2, rng) if rng.random() < crossover_rate else p1
                child = mutate(child, space, rng, mutation_rate, ctx.timeframes, reg)
                new_pop.append(child)

        pop = new_pop

        if gen % max(1, generations // 10) == 0 or gen == generations - 1:
            save_json(out_dir / "history.json", history)
            top = []
            for g, f, extra in scored[:topk]:
                top.append({
                    "fitness": float(f),
                    "spec": extra["spec"],
                    "metrics": extra["metrics"],
                })
            save_json(out_dir / "topk.json", top)

    save_json(out_dir / "history.json", history)
    top = []
    for g, f, extra in scored[:topk]:
        top.append({
            "fitness": float(f),
            "spec": extra["spec"],
            "metrics": extra["metrics"],
        })
    save_json(out_dir / "topk.json", top)

    # Export python trade lists for parity validation (Top-K on a fixed symbol).
    try:
        parity_symbol = ctx.parity_symbol
        if parity_symbol:
            parity_symbol = str(parity_symbol).upper()
            if parity_symbol not in ctx.data:
                parity_symbol = None

        # fallback: prefer a large/liquid symbol if present
        if parity_symbol is None:
            preferred = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            for s in preferred:
                if s in ctx.data:
                    parity_symbol = s
                    break
        if parity_symbol is None:
            parity_symbol = sorted(ctx.data.keys())[0]
        parity_dir = ensure_dir(out_dir / "python_trades")
        for rank, item in enumerate(top, start=1):
            spec = strategy_spec_from_dict(item["spec"])
            trigger_df = ctx.data[parity_symbol][spec.tf_trigger]
            anchor_df = ctx.data[parity_symbol][spec.tf_anchor]
            bt = backtest_mtf_close_to_close(
                trigger_df=trigger_df,
                anchor_df=anchor_df,
                symbol=parity_symbol,
                spec=spec,
                commission_bps=float(ctx.commission_bps),
                slippage_ticks=int(ctx.slippage_ticks),
                tick_size=ctx.tick_size_by_symbol.get(parity_symbol),
                fill_mode=str(ctx.fill_mode),
                notional=float(ctx.notional),
                indicator_registry=ctx.indicators,
                use_numba=bool(ctx.use_numba),
            )
            payload = {
                "rank": rank,
                "symbol": parity_symbol,
                "spec": item["spec"],
                "metrics": bt.metrics,
                "trades": [_trade_to_dict(t) for t in bt.trades],
            }
            save_json(parity_dir / f"strategy_{rank:03d}.json", payload)
    except Exception:
        # Non-fatal; GA results still valid.
        pass

    # write persistent cache if requested
    if 'cache_file' in locals() and locals().get('cache_file'):
        try:
            with open(locals().get('cache_file'), 'wb') as _f:
                pickle.dump(scored_cache, _f)
        except Exception:
            pass

    return out_dir