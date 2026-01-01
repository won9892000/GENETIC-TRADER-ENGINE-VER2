"""Small harness to run a short GA for profiling/debugging."""
from pathlib import Path
from ga_trader.config import Config
from ga_trader.utils import load_json
from ga_trader.ga.runner import run_ga, EvalContext
import pandas as pd

cfg = Config.load('configs/config.quick_test.yaml')
u = load_json(Path('configs/universe_quick.json'))
raw_symbols = u.get('symbols', [])
symbols = [s if isinstance(s, str) else str(s.get('symbol')) for s in raw_symbols]

# load minimal data for first symbol only to make it fast
s = symbols[0]
store_dir = Path(cfg.raw.get('data', {}).get('store_dir', 'data'))
# use only 5m timeframe
tfs = ['5m']
data = {s: {}}
for tf in tfs:
    path = store_dir / f"{s}_{tf}.csv"
    df = pd.read_csv(path)
    data[s][tf] = df

ctx = EvalContext(
    data=data,
    commission_bps=float(cfg.raw.get('fees', {}).get('commission_bps', 0.0)),
    slippage_ticks=int(cfg.raw.get('execution', {}).get('slippage_ticks', 0)),
    tick_size_by_symbol={},
    fill_mode=str(cfg.raw.get('execution', {}).get('fill', 'same_bar_close')),
    notional=float(cfg.raw.get('risk', {}).get('fixed_notional', 100.0)),
    split=cfg.raw.get('data', {}).get('split', {'is':0.6,'oos':0.2,'hold':0.2}),
    timeframes=tfs,
    indicators=None,
    parity_symbol=cfg.parity_symbol,
    use_numba=False,
)

space = cfg.raw.get('strategy_space', {})
fg = cfg.raw.get('fitness', {})

out_dir = Path('runs/profile_run')

# Run tiny GA
run_ga(space=space, ctx=ctx, fitness_cfg=fg, population=6, generations=2, elite_ratio=0.1, tournament_k=3, crossover_rate=0.7, mutation_rate=0.2, rng_seed=42, out_dir=out_dir, topk=3, workers=1, cache_file=None, use_processes=False)
print('Profile run complete')
