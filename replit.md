# GA Trading Strategy Engine

## Overview
A genetic algorithm-based trading strategy optimization engine with a Streamlit web interface. The system evolves trading strategies using evolutionary algorithms to find optimal parameter combinations.

## Project Structure
```
.
├── app.py                  # Streamlit web interface
├── ga_trader/              # Core GA trading engine module
│   ├── backtest/           # Backtesting engine
│   │   ├── engine.py       # Main backtest logic
│   │   └── export.py       # Export utilities
│   ├── commands/           # CLI commands
│   ├── fitness/            # Fitness evaluation
│   │   ├── evaluator.py    # Fitness function
│   │   └── metrics.py      # Performance metrics
│   ├── ga/                 # Genetic algorithm core
│   │   ├── genome.py       # Strategy genome representation
│   │   ├── nsga2.py        # NSGA-II multi-objective selection
│   │   └── runner.py       # Main GA runner
│   ├── indicators/         # Technical indicators
│   │   ├── registry.py     # Indicator registry
│   │   └── spec.py         # Indicator specifications
│   ├── strategy/           # Strategy specifications
│   │   └── spec.py         # StrategySpec dataclass
│   ├── ta/                 # Technical analysis functions
│   │   └── core.py         # EMA, RSI, ATR, etc.
│   └── utils.py            # Utility functions
├── indicators/             # Indicator YAML specs
│   └── specs/              # Individual indicator definitions
├── data/                   # OHLCV CSV data files
├── runs/                   # GA run output directory
└── configs/                # Configuration files
```

## Key Features
- Genetic algorithm optimization for trading strategies
- Multi-timeframe backtesting (MTF)
- EMA crossover trend detection
- RSI momentum filtering
- ATR-based stop-loss and take-profit
- Z-score filtering
- Custom indicator support via YAML specs

## Running the Application
The app runs on port 5000 using Streamlit:
```bash
streamlit run app.py --server.port 5000
```

## Strategy Parameters
- **EMA Fast/Slow**: Trend detection via exponential moving average crossover
- **RSI Length**: RSI indicator period
- **RSI Long/Short Thresholds**: Entry signal thresholds
- **ATR Length**: Average True Range period
- **SL/TP ATR Multipliers**: Stop-loss and take-profit as ATR multiples
- **Z-Score Length/Entry**: Optional momentum filter

## GA Configuration
- **Population**: Number of strategies per generation
- **Generations**: Evolution cycles
- **Elite Ratio**: Top performers kept unchanged
- **Crossover Rate**: Probability of gene crossover
- **Mutation Rate**: Probability of random mutations

## Data Formats
- OHLCV CSV files with columns: timestamp, open, high, low, close, volume
- Indicator specs in YAML format

## Recent Changes
- 2024-12-31: Initial setup with Streamlit interface
- 2024-12-31: Fixed missing PyYAML dependency
- 2024-12-31: Successful GA optimization run verified
