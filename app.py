import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ga_trader.ga.runner import run_ga, EvalContext
from ga_trader.examples.synthetic import make_synthetic_ohlcv
from ga_trader.utils import ensure_dir
from ga_trader.indicators.registry import IndicatorRegistry

st.set_page_config(
    page_title="GA Trading Strategy Engine",
    page_icon="📈",
    layout="wide"
)

st.title("GA Trading Strategy Engine")
st.markdown("유전 알고리즘을 활용한 트레이딩 전략 최적화 엔진")

with st.sidebar:
    st.header("Settings")
    
    data_source = st.selectbox(
        "Data Source",
        ["Synthetic (Demo)", "Real Data (CSV)"],
        help="Choose data source for backtesting"
    )
    
    st.subheader("GA Parameters")
    population = st.slider("Population Size", 10, 100, 20, help="Number of strategies per generation")
    generations = st.slider("Generations", 1, 50, 5, help="Number of evolution cycles")
    elite_ratio = st.slider("Elite Ratio", 0.05, 0.30, 0.10, help="Top performers to keep")
    tournament_k = st.slider("Tournament Size", 2, 10, 3, help="Selection tournament size")
    crossover_rate = st.slider("Crossover Rate", 0.5, 1.0, 0.70, help="Probability of crossover")
    mutation_rate = st.slider("Mutation Rate", 0.05, 0.50, 0.20, help="Probability of mutation")
    rng_seed = st.number_input("Random Seed", 1, 1000, 7, help="Reproducibility seed")
    topk = st.slider("Top K Results", 3, 20, 5, help="Number of top strategies to save")

    st.subheader("Strategy Space")
    ema_fast_range = st.slider("EMA Fast Range", 5, 50, (5, 30))
    ema_slow_range = st.slider("EMA Slow Range", 20, 150, (20, 80))
    rsi_len_range = st.slider("RSI Length Range", 5, 50, (6, 30))
    atr_len_range = st.slider("ATR Length Range", 5, 50, (7, 28))

    st.subheader("Backtest Settings")
    commission_bps = st.number_input("Commission (bps)", 0.0, 20.0, 4.0)
    notional = st.number_input("Notional Size ($)", 10.0, 10000.0, 100.0)

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Run GA Optimization", type="primary", use_container_width=True):
        with st.spinner("Running genetic algorithm optimization..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if data_source == "Synthetic (Demo)":
                status_text.text("Generating synthetic data...")
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
                tick_sizes = {"SYN1USDT": 0.01, "SYN2USDT": 0.01}
            else:
                status_text.text("Loading real data from CSV files...")
                data_dir = Path("data")
                data = {}
                tick_sizes = {}
                if data_dir.exists():
                    csv_files = list(data_dir.glob("*.csv"))
                    for csv_file in csv_files[:10]:
                        try:
                            name = csv_file.stem
                            parts = name.rsplit("_", 1)
                            if len(parts) == 2:
                                symbol, tf = parts
                                df = pd.read_csv(csv_file)
                                required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
                                missing_cols = required_cols - set(df.columns)
                                if missing_cols:
                                    st.warning(
                                        f"Skipping {csv_file.name}: missing columns {sorted(missing_cols)}"
                                    )
                                    continue
                                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                                df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                                if df.empty:
                                    st.warning(f"Skipping {csv_file.name}: no valid timestamp rows found")
                                    continue
                                if symbol not in data:
                                    data[symbol] = {}
                                data[symbol][tf] = df
                                if symbol not in tick_sizes:
                                    tick_sizes[symbol] = 0.01
                        except Exception as e:
                            st.warning(f"Failed to load {csv_file.name}: {e}")
                
                if not data:
                    st.error("No data files found in 'data' folder. Using synthetic data instead.")
                    data = {
                        "SYN1USDT": {
                            "5m": make_synthetic_ohlcv(n=1200, minutes=5, seed=1),
                            "15m": make_synthetic_ohlcv(n=600, minutes=15, seed=11),
                        },
                    }
                    tick_sizes = {"SYN1USDT": 0.01}

            progress_bar.progress(10)
            status_text.text("Loading indicator registry...")
            
            indicators_root = Path("indicators")
            reg = IndicatorRegistry.from_root(indicators_root)
            
            progress_bar.progress(20)
            status_text.text("Setting up evaluation context...")

            ctx = EvalContext(
                data=data,
                commission_bps=commission_bps,
                slippage_ticks=0,
                tick_size_by_symbol=tick_sizes,
                fill_mode="same_bar_close",
                notional=notional,
                split={"is": 0.6, "oos": 0.2, "hold": 0.2},
                timeframes=["5m", "15m"],
                indicators=reg,
            )

            space = {
                "ema_fast": list(ema_fast_range),
                "ema_slow": list(ema_slow_range),
                "rsi_len": list(rsi_len_range),
                "rsi_long": [50, 70],
                "rsi_short": [30, 50],
                "atr_len": list(atr_len_range),
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

            progress_bar.progress(30)
            status_text.text("Running genetic algorithm optimization...")

            out_dir = Path("runs/streamlit_run")
            ensure_dir(out_dir)

            try:
                run_ga(
                    space=space,
                    ctx=ctx,
                    fitness_cfg=fitness_cfg,
                    population=population,
                    generations=generations,
                    elite_ratio=elite_ratio,
                    tournament_k=tournament_k,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    rng_seed=rng_seed,
                    out_dir=out_dir,
                    topk=topk,
                )
                progress_bar.progress(100)
                status_text.text("Optimization complete!")
                st.success(f"GA optimization completed! Results saved to {out_dir}")
                st.session_state.run_completed = True
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with col2:
    st.subheader("Quick Info")
    st.info("""
    **How it works:**
    1. Configure parameters in sidebar
    2. Click 'Run GA Optimization'
    3. View results below
    
    **Strategy Components:**
    - EMA crossover for trend
    - RSI for entry signals
    - ATR-based stops
    """)

st.markdown("---")

results_dir = Path("runs/streamlit_run")
if results_dir.exists():
    st.header("Results")
    
    topk_file = results_dir / "topk.json"
    history_file = results_dir / "history.json"
    
    if topk_file.exists():
        with open(topk_file) as f:
            topk_results = json.load(f)
        
        st.subheader("Top Strategies")
        
        for i, result in enumerate(topk_results):
            with st.expander(f"Strategy #{i+1} - Fitness: {result['fitness']:.4f}", expanded=(i==0)):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Strategy Parameters:**")
                    spec = result['spec']
                    params_df = pd.DataFrame({
                        "Parameter": [
                            "Anchor Timeframe",
                            "Trigger Timeframe",
                            "EMA Fast",
                            "EMA Slow",
                            "RSI Length",
                            "RSI Long Threshold",
                            "RSI Short Threshold",
                            "ATR Length",
                            "Stop Loss (ATR mult)",
                            "Take Profit (ATR mult)",
                        ],
                        "Value": [
                            str(spec.get('tf_anchor', 'N/A')),
                            str(spec.get('tf_trigger', 'N/A')),
                            str(spec.get('ema_fast', 'N/A')),
                            str(spec.get('ema_slow', 'N/A')),
                            str(spec.get('rsi_len', 'N/A')),
                            f"{spec.get('rsi_long', 0):.2f}",
                            f"{spec.get('rsi_short', 0):.2f}",
                            str(spec.get('atr_len', 'N/A')),
                            f"{spec.get('sl_atr', 0):.2f}",
                            f"{spec.get('tp_atr', 0):.2f}",
                        ]
                    })
                    st.dataframe(params_df, hide_index=True)
                
                with col_b:
                    st.markdown("**Performance Metrics:**")
                    metrics = result['metrics']
                    metrics_df = pd.DataFrame({
                        "Metric": [
                            "OOS Expectancy",
                            "OOS Profit Factor",
                            "OOS Win Rate",
                            "OOS Trades/Month",
                            "OOS Max Drawdown",
                            "IS Expectancy",
                            "IS Profit Factor",
                            "IS Win Rate",
                        ],
                        "Value": [
                            f"{metrics.get('oos_expectancy', 0):.4f}",
                            f"{metrics.get('oos_profit_factor', 0):.2f}",
                            f"{metrics.get('oos_win_rate', 0)*100:.1f}%",
                            f"{metrics.get('oos_trades_per_month', 0):.1f}",
                            f"{metrics.get('oos_max_drawdown', 0)*100:.2f}%",
                            f"{metrics.get('is_expectancy', 0):.4f}",
                            f"{metrics.get('is_profit_factor', 0):.2f}",
                            f"{metrics.get('is_win_rate', 0)*100:.1f}%",
                        ]
                    })
                    st.dataframe(metrics_df, hide_index=True)
    
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        
        st.subheader("Evolution Progress")
        
        fitness_history = [h['best_fitness'] for h in history]
        generations_list = [h['generation'] for h in history]
        
        chart_df = pd.DataFrame({
            "Generation": generations_list,
            "Best Fitness": fitness_history
        })
        st.line_chart(chart_df.set_index("Generation"))
        
        st.subheader("Generation Details")
        gen_df = pd.DataFrame([
            {
                "Generation": h['generation'],
                "Best Fitness": f"{h['best_fitness']:.4f}",
                "OOS Expectancy": f"{h['best_metrics'].get('oos_expectancy', 0):.4f}",
                "OOS Win Rate": f"{h['best_metrics'].get('oos_win_rate', 0)*100:.1f}%",
            }
            for h in history
        ])
        st.dataframe(gen_df, hide_index=True)

else:
    st.info("No results yet. Run GA optimization to see results here.")

st.markdown("---")
st.caption("GA Trading Strategy Engine - Powered by Genetic Algorithm Optimization")
