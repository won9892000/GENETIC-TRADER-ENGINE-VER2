# Optimization summary

## What I changed ✅
- Vectorized rolling statistics (SMA/STDEV/Z-score) with O(n) cumulative-sum approaches to avoid per-window Python loops.
- Replaced anchor->trigger alignment with `np.searchsorted` to eliminate Python scanning loops.
- Added threaded evaluation of genomes (`workers` option / `GA_WORKERS` env var) to use multiple CPU cores without heavy pickling costs.
- Added an optional persistent cache (`--cache-file`) to store evaluated spec results across runs.
- Added a small benchmark harness (`tools/run_bench.py`) and a pytest performance test (`tests/test_performance.py`).

## How to use
- Run with multiple workers (recommended):
  - CLI: `python -m ga_trader.cli ga run -c configs/config.example.yaml -u runs/universe.json --workers 8`
  - Or set env var `GA_WORKERS=8` to control default worker count.
- Use multiprocessing instead of threads for isolated worker processes (trade-off: higher start-up/pickling cost):
  - `--use-processes` to enable process pool evaluation.

- Use numba JIT for the inner backtest loop (optional):
  - Install optional dependency: `pip install .[numba]`
  - Enable with CLI: `--use-numba` (will be used only if numba is installed).

- Use persistent cache to speed up repeated runs:
  - `--cache-file runs/myrun/scored_cache.pkl` will persist evaluated specs between runs.

## Notes & next steps
- Implemented `numba` JIT path for the core entry/exit index detection. This gives additional speed when working with large single-threaded runs and numba is available.
- CI performance gating added (lightweight bench) at `.github/workflows/perf.yml` to catch regressions.

## Integration run (FinRL + numba) ✅

- I installed optional packages: `numba`, `finrl`, `yfinance`, `ta`, `pandas_ta` into the devcontainer environment.
- Generated FinRL-inspired indicator specs with `ga_trader.providers.finrl.generate_indicator_specs()` (wrote 7 specs into `indicators/specs`).
- Ran unit tests and perf tests (all passed locally).
- Ran the synthetic benchmark harness: output written to `runs/synth_bench/topk.json`.
- Ran a quick real-data GA with FinRL indicators and numba enabled:
  - Command used:
    - `PYTHONPATH=$PWD .venv/bin/python -m ga_trader.cli ga run -c configs/config.quick_test.yaml -u configs/universe_quick.json --indicators_root indicators --use-numba --workers 2`
  - Result run directory: `runs/quick_test`
  - Top-1 fitness: `0.5035` (see `runs/quick_test/topk.json`)

These results confirm the FinRL provider and numba JIT code path work in this environment and the GA can run end-to-end using the new indicator specs.
