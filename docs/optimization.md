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

- Use persistent cache to speed up repeated runs:
  - `--cache-file runs/myrun/scored_cache.pkl` will persist evaluated specs between runs.

## Notes & next steps
- Consider adding optional `numba` JIT paths for the inner backtest loop to further speed sequential logic.
- Add CI performance gating (e.g., a lightweight benchmark smoke-test) to catch regressions.

