from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from ga_trader.fitness.metrics import normalize_01

@dataclass
class FitnessResult:
    fitness: float
    passed: bool
    details: Dict[str, float]

def _apply_constraints(metrics: Dict[str, float], constraints: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
    passed = True
    details = {}
    # convention:
    # min_* => metric >= value
    # max_* => metric <= value
    for k, v in constraints.items():
        if k.startswith("min_"):
            m = k[len("min_"):]
            ok = metrics.get(m, 0.0) >= float(v)
        elif k.startswith("max_"):
            m = k[len("max_"):]
            ok = metrics.get(m, 0.0) <= float(v)
        else:
            continue
        details[f"constraint_{k}"] = 1.0 if ok else 0.0
        if not ok:
            passed = False
    return passed, details

def weighted_sum(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    # metric-specific normalization defaults (MVP heuristics)
    # You can refine these per your preference.
    normed = {}
    normed["oos_expectancy"] = normalize_01(metrics.get("oos_expectancy", 0.0), -0.5, 0.5)
    normed["oos_profit_factor"] = normalize_01(metrics.get("oos_profit_factor", 0.0), 0.8, 2.5)
    normed["oos_win_rate"] = normalize_01(metrics.get("oos_win_rate", 0.0), 0.3, 0.8)
    normed["oos_trades_per_month"] = normalize_01(metrics.get("oos_trades_per_month", 0.0), 0.0, 300.0)
    # drawdown is penalty: smaller is better
    normed["oos_max_drawdown"] = 1.0 - normalize_01(metrics.get("oos_max_drawdown", 1.0), 0.0, 0.6)
    # IS-OOS gap penalty: smaller better
    normed["is_oos_gap"] = 1.0 - normalize_01(metrics.get("is_oos_gap", 1.0), 0.0, 1.0)

    s = 0.0
    for k, w in weights.items():
        if k not in normed:
            # allow raw metrics if user wants (assumed already 0..1)
            s += float(w) * float(metrics.get(k, 0.0))
        else:
            s += float(w) * float(normed[k])
    return float(s)

def evaluate_fitness(metrics: Dict[str, float], mode: str, weights: Dict[str, float], constraints: Dict[str, float]) -> FitnessResult:
    passed, cdetails = _apply_constraints(metrics, constraints or {})
    if mode == "constraints":
        # fitness = number of constraints satisfied + small tie-breaker on expectancy
        fitness = float(sum(cdetails.values())) + 0.001 * float(metrics.get("oos_expectancy", 0.0))
        return FitnessResult(fitness=fitness, passed=passed, details={**metrics, **cdetails})

    if mode == "weighted_sum":
        fitness = weighted_sum(metrics, weights or {})
        if not passed:
            fitness -= 10.0  # hard penalty
        return FitnessResult(fitness=fitness, passed=passed, details={**metrics, **cdetails})

    if mode == "pareto":
        # MVP: treat as weighted_sum unless user plugs NSGA-II; keep interface stable.
        fitness = weighted_sum(metrics, weights or {})
        if not passed:
            fitness -= 10.0
        return FitnessResult(fitness=fitness, passed=passed, details={**metrics, **cdetails})

    raise ValueError(f"Unknown fitness mode: {mode}")
