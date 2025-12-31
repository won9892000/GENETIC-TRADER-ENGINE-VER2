from pathlib import Path

from ga_trader.ga.runner import run_ga, EvalContext
from ga_trader.examples.synthetic import make_synthetic_ohlcv
from ga_trader.utils import ensure_dir
from ga_trader.indicators.registry import IndicatorRegistry

def main():
    # Synthetic universe with 2 symbols and 2 timeframes (fast smoke test)
    data = {
        "SYN1USDT": {
            "5m": make_synthetic_ohlcv(n=1200, minutes=5, seed=1),
            "15m": make_synthetic_ohlcv(n=600, minutes=15, seed=11),
        },
        "SYN2USDT": {
            "5m": make_synthetic_ohlcv(n=1200, minutes=5, seed=2),
            "15m": make_synthetic_ohlcv(n=600, minutes=15, seed=22),
        },
    }

    indicators_root = Path("indicators")
    reg = IndicatorRegistry.from_root(indicators_root)

    ctx = EvalContext(
        data=data,
        commission_bps=4.0,
        slippage_ticks=0,
        tick_size_by_symbol={"SYN1USDT": 0.01, "SYN2USDT": 0.01},
        fill_mode="same_bar_close",  # closest to Pine process_orders_on_close=true
        notional=100.0,
        split={"is": 0.6, "oos": 0.2, "hold": 0.2},
        timeframes=["5m", "15m"],
        indicators=reg,
    )

    space = {
        "ema_fast": [5, 30],
        "ema_slow": [20, 80],
        "rsi_len": [6, 30],
        "rsi_long": [50, 70],
        "rsi_short": [30, 50],
        "atr_len": [7, 28],
        "sl_atr": [0.6, 2.0],
        "tp_atr": [0.6, 2.5],
        "z_len": [0, 40],
        "z_entry": [0.0, 2.0],
        "max_extra_filters": 1,
    }

    fitness_cfg = {
        "mode": "weighted_sum",
        "weights": {
            "oos_expectancy": 2.0,
            "oos_profit_factor": 0.5,
            "oos_win_rate": 0.5,
            "oos_trades_per_month": 0.2,
            "oos_max_drawdown": -1.0,
            "is_oos_gap": -0.5,
        },
        "constraints": {
            "oos_trades_per_month": [1, 300],
        },
    }

    out_dir = Path("runs/synth_demo")
    ensure_dir(out_dir)
    run_ga(
        space=space,
        ctx=ctx,
        fitness_cfg=fitness_cfg,
        population=20,
        generations=5,
        elite_ratio=0.10,
        tournament_k=3,
        crossover_rate=0.70,
        mutation_rate=0.20,
        rng_seed=7,
        out_dir=out_dir,
        topk=5,
    )

    print("Done. See runs/synth_demo/topk.json")
    print("You can export Pine via:")
    print("  python -m ga_trader export pine -c configs/config.example.yaml --run_dir runs/synth_demo --topk 5 --out pine_out")

if __name__ == "__main__":
    main()
