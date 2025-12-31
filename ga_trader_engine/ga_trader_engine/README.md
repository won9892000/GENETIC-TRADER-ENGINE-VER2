# GA Trader Engine (Python GA + Pine Export)

This is a working MVP of the engine described in the spec:
- Universe = Top 24h quote-volume symbols on Binance USDT-M Perp (configurable top N)
- Timeframes: 3m / 5m / 15m
- Execution: bar-close model, default **next_bar_close** fills
- Fees: configurable commission (bps) in `config.yaml`
- Fitness: user-configurable (weighted sum + constraints; Pareto scaffold included)
- Output: Top-K strategies + auto-generated TradingView Pine v5 strategies
- Parity: a helper that can compare Python trades vs TradingView exported trades CSV (best-effort parser)

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure
Copy and edit:
```bash
cp configs/config.example.yaml config.yaml
```

### 3) Build universe (Top market-cap coins + Binance perp intersection)
```bash
python -m ga_trader universe build --config config.yaml --out runs/universe.json
```

### 4) Fetch data (Binance Futures klines)
```bash
python -m ga_trader data fetch --config config.yaml --universe runs/universe.json
```

### 5) Run GA
```bash
python -m ga_trader ga run --config config.yaml --universe runs/universe.json
```

### 6) Export top strategies to Pine
```bash
python -m ga_trader export pine --config config.yaml --run_dir runs/latest --topk 20 --out pine_out
```

## Notes on parity with TradingView
Perfect parity requires:
- identical OHLCV candles (same exchange/symbol mapping and timestamps)
- identical indicator definitions (this MVP implements Pine-equivalent EMA/RMA/RSI/ATR/STDEV with explicit seeding rules)
- identical execution assumptions (this MVP is close-to-close; TV often simulates intrabar order fills if enabled)

Recommended for parity:
- set TV strategy to **process orders on close** and **disable intrabar fill assumptions**
- use the same fee settings

## Indicators (open-source Pine as gene pool)

This version adds an **indicator registry** under `indicators/`.
- `indicators/specs/*.yaml` define indicator parameters, outputs, Python implementation, and a Pine snippet.
- GA can attach up to `strategy_space.max_extra_filters` indicator-threshold filters to entries.
- Add your own open-source Pine indicators via `python -m ga_trader indicators harvest --pine <file.pine>` then fill the spec + implement Python equivalent.

## What is implemented in this MVP
- Core TA: SMA, EMA, RMA (Wilder), RSI, ATR, STDEV, Z-score
- Strategy family: EMA trend filter + RSI trigger + optional z-score filter; SL/TP in ATR multiples
- GA: tournament selection, uniform crossover, bounded mutation; multi-symbol + multi-timeframe evaluation
- Fitness: weighted sum + constraints; optional IS/OOS split penalty

## Extending
Add more indicator families by:
1) implementing a Pine-equivalent function in `ga_trader/ta/core.py`
2) adding a gene and code-gen mapping in `ga_trader/ga/genome.py` and `ga_trader/export/pine.py`

