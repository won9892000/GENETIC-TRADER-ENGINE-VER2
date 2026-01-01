"""Lightweight Streamlit UI to run the GA engine interactively.

Usage (from repo root):
  PYTHONPATH=$PWD .venv/bin/python -m streamlit run apps/streamlit_app.py

This app is intentionally minimal — it exposes common CLI options and runs the
GA synchronously, showing the resulting topk.json when finished. For long
runs, prefer running the CLI in a terminal or running this in a detached
environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

from ga_trader.config import Config
from ga_trader.utils import load_json
from ga_trader.ga.runner import run_ga, EvalContext
from ga_trader.indicators.registry import IndicatorRegistry

st.set_page_config(page_title="GA Trader Engine — Streamlit", layout="wide")

ROOT = Path.cwd()
CONFIG_DIR = ROOT / "configs"
IND_DIR = ROOT / "indicators"


def list_configs():
    return sorted([p.name for p in CONFIG_DIR.glob("*.yaml")])


def list_universes():
    return sorted([p.name for p in CONFIG_DIR.glob("*.json")])


st.title("GA Trader Engine — Interactive Runner")

with st.sidebar:
    st.header("Run settings")
    cfg_file = st.selectbox("Config YAML", options=list_configs())
    uni_file = st.selectbox("Universe JSON", options=list_universes())
    indicators_root = st.text_input("Indicators root", value=str(IND_DIR))
    workers = st.number_input("Workers (None = auto)", min_value=0, value=2, step=1)
    use_processes = st.checkbox("Use processes (multiprocessing)", value=False)
    use_numba = st.checkbox("Enable numba JIT (if installed)", value=False)
    run_name = st.text_input("Run name", value="streamlit_run")
    cache_file = st.text_input("Optional cache file", value="")
    st.markdown("---")
    if st.button("Generate FinRL indicator specs"):
        try:
            from ga_trader.providers.finrl import generate_indicator_specs

            n = generate_indicator_specs(indicators_root)
            st.success(f"Wrote {n} specs to {indicators_root}/specs")
        except Exception as exc:
            st.error(f"Failed to generate specs: {exc}")


st.subheader("Selected config & universe")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Config**")
    cfg_path = CONFIG_DIR / cfg_file
    try:
        st.code(cfg_path.read_text(encoding="utf-8"), language="yaml")
    except Exception:
        st.warning("Could not read config file")
with col2:
    st.markdown("**Universe**")
    uni_path = CONFIG_DIR / uni_file
    try:
        u = load_json(uni_path)
        st.json(u)
    except Exception:
        st.warning("Could not read universe file")

st.markdown("---")

st.header("Run GA")

if st.button("Run GA now"):
    # Load config / universe and prepare data similar to CLI
    try:
        cfg = Config.load(cfg_path)
        u = load_json(uni_path)

        raw_symbols = u.get("symbols", [])
        symbols = []
        tick_size_by_symbol = {}
        for it in raw_symbols:
            if isinstance(it, dict):
                sym = str(it.get("symbol"))
                symbols.append(sym)
                if "tick_size" in it and it["tick_size"] is not None:
                    tick_size_by_symbol[sym] = float(it["tick_size"])
            else:
                symbols.append(str(it))

        if not symbols:
            st.error("Universe has no symbols")
            st.stop()

        dcfg = cfg.raw.get("data", {})
        tfs = list(dcfg.get("timeframes", ["5m"]))
        store_dir = Path(dcfg.get("store_dir", "data"))

        data: dict[str, dict[str, pd.DataFrame]] = {}
        missing = []
        for sym in symbols:
            s = str(sym)
            tmap = {}
            for tf in tfs:
                path = store_dir / f"{s}_{tf}.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                df = pd.read_csv(path)
                need = {"timestamp", "open", "high", "low", "close", "volume"}
                if not need.issubset(df.columns):
                    st.error(f"{path} missing columns: {need - set(df.columns)}")
                    st.stop()
                tmap[tf] = df
            if tmap:
                data[s] = tmap

        if missing:
            st.warning("Missing data files (run data fetch first):")
            for m in missing[:10]:
                st.text(m)
            st.stop()

        gcfg = cfg.raw.get("ga", {})
        fcfg = cfg.raw.get("fitness", {})
        space = cfg.raw.get("strategy_space", {})

        run_root = Path(cfg.raw.get("project", {}).get("run_root", "runs"))
        out_dir = run_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        reg = (
            IndicatorRegistry.from_root(Path(indicators_root))
            if Path(indicators_root).exists()
            else None
        )

        ctx = EvalContext(
            data=data,
            commission_bps=float(cfg.raw.get("fees", {}).get("commission_bps", 0.0)),
            slippage_ticks=int(cfg.raw.get("execution", {}).get("slippage_ticks", 0)),
            tick_size_by_symbol=tick_size_by_symbol,
            fill_mode=str(cfg.raw.get("execution", {}).get("fill", "same_bar_close")),
            notional=float(cfg.raw.get("risk", {}).get("fixed_notional", 100.0)),
            split=cfg.raw.get("data", {}).get("split", {"is": 0.6, "oos": 0.2, "hold": 0.2}),
            timeframes=tfs,
            indicators=reg,
            parity_symbol=cfg.parity_symbol,
            use_numba=use_numba,
        )

        st.info("Starting GA (this will run in-process). This may take a while for large configs.")
        with st.spinner("Running GA..."):
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
                workers=(None if workers == 0 else int(workers)),
                cache_file=(cache_file or None),
                use_processes=use_processes,
            )

        st.success(f"GA finished. Results written to: {out_dir}")
        topk_path = out_dir / "topk.json"
        if topk_path.exists():
            topk = json.loads(topk_path.read_text(encoding="utf-8"))
            st.subheader("Top-k strategies")
            st.table(pd.DataFrame([{"fitness": b["fitness"], **b["spec"]} for b in topk]))
        else:
            st.warning("No topk.json found in run directory")

    except Exception as exc:
        st.exception(exc)
