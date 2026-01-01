import time
import pytest
from pathlib import Path
from ga_trader.ga.runner import run_ga, EvalContext
from ga_trader.examples.synthetic import make_synthetic_ohlcv
from ga_trader.utils import ensure_dir
from ga_trader.indicators.registry import IndicatorRegistry


def make_ctx():
    data = {
        "SYN1USDT": {
            "5m": make_synthetic_ohlcv(n=800, minutes=5, seed=1),
            "15m": make_synthetic_ohlcv(n=400, minutes=15, seed=11),
        }
    }
    reg = IndicatorRegistry.from_root(Path("indicators"))
    ctx = EvalContext(
        data=data,
        commission_bps=4.0,
        slippage_ticks=0,
        tick_size_by_symbol={"SYN1USDT": 0.01},
        fill_mode="same_bar_close",
        notional=100.0,
        split={"is": 0.6, "oos": 0.2, "hold": 0.2},
        timeframes=["5m", "15m"],
        indicators=reg,
    )
    return ctx


def test_multiprocess_runs(tmp_path):
    out_dir = tmp_path / "mp_run"
    ensure_dir(out_dir)
    ctx = make_ctx()
    space = {"ema_fast": [5, 20], "ema_slow": [20, 60], "rsi_len": [6, 16], "rsi_long": [50, 70], "rsi_short": [30, 50], "atr_len": [7, 14], "sl_atr": [0.6, 2.0], "tp_atr": [0.6, 2.5], "z_len": [0, 40], "z_entry": [0.0, 2.0]}
    fitness_cfg = {"mode": "weighted_sum", "weights": {"oos_expectancy": 2.0}}

    start = time.perf_counter()
    run_ga(
        space=space,
        ctx=ctx,
        fitness_cfg=fitness_cfg,
        population=12,
        generations=2,
        elite_ratio=0.1,
        tournament_k=3,
        crossover_rate=0.7,
        mutation_rate=0.2,
        rng_seed=7,
        out_dir=out_dir,
        topk=3,
        workers=2,
        use_processes=True,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Multiprocess GA run too slow: {elapsed}s"


@pytest.mark.skipif(not hasattr(__import__('ga_trader.backtest.engine'), '_NUMBA_AVAILABLE') or not __import__('ga_trader.backtest.engine')._NUMBA_AVAILABLE,
                    reason="numba unavailable")
def test_numba_path(tmp_path):
    out_dir = tmp_path / "nb_run"
    ensure_dir(out_dir)
    ctx = make_ctx()
    ctx.use_numba = True
    space = {"ema_fast": [5, 20], "ema_slow": [20, 60], "rsi_len": [6, 16], "rsi_long": [50, 70], "rsi_short": [30, 50], "atr_len": [7, 14], "sl_atr": [0.6, 2.0], "tp_atr": [0.6, 2.5], "z_len": [0, 40], "z_entry": [0.0, 2.0]}
    fitness_cfg = {"mode": "weighted_sum", "weights": {"oos_expectancy": 2.0}}

    run_ga(
        space=space,
        ctx=ctx,
        fitness_cfg=fitness_cfg,
        population=12,
        generations=2,
        elite_ratio=0.1,
        tournament_k=3,
        crossover_rate=0.7,
        mutation_rate=0.2,
        rng_seed=7,
        out_dir=out_dir,
        topk=3,
        workers=1,
    )
