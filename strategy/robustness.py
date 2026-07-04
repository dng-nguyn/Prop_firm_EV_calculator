"""
Robustness Testing for Trade Strategies
========================================
Validates that strategy performance is not overfit.

Tests:
1. Entry delay sensitivity (shift entry by 1-N bars)
2. Walk-forward analysis (rolling IS/OOS windows)
3. Price perturbation (add noise to entry/exit prices)
4. Multi-timeframe consistency (same logic on 1m, 5m, 15m bars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.vwap_strategy import (
    StrategyParams,
    generate_trades,
    load_nq_data,
    resample_to_bars,
    trades_to_dataframe,
)

log = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Results of robustness testing."""
    baseline_pnl: float
    baseline_win_rate: float
    baseline_trades: int
    
    # Delay test
    delay_1_pnl: float        # PnL with 1-bar delay
    delay_1_degradation: float # % PnL lost vs baseline
    delay_2_pnl: float
    delay_2_degradation: float
    
    # Walk-forward
    wf_oos_pnl: float         # Average OOS PnL across folds
    wf_oos_win_rate: float
    wf_is_pnl: float          # Average IS PnL
    wf_efficiency: float      # OOS/IS ratio (want > 0.5)
    
    # Perturbation
    perturb_1tick_pnl: float
    perturb_1tick_degradation: float
    
    # Multi-timeframe
    tf_5m_pnl: float
    tf_15m_pnl: float
    tf_consistency: float      # % of timeframes that are profitable
    
    # Overall
    robust: bool               # Passes all robustness checks
    issues: list[str]


def test_entry_delay(
    df: pd.DataFrame,
    params: StrategyParams,
    bar_minutes: int = 1,
) -> dict:
    """Test PnL degradation with entry delays of 1, 2, 3 bars."""
    from strategy.vwap_strategy import generate_trades
    
    results = {}
    baseline_trades = generate_trades(df, params, bar_minutes)
    baseline_pnl = sum(t.pnl for t in baseline_trades)
    results["baseline"] = {"pnl": baseline_pnl, "n_trades": len(baseline_trades)}
    
    for delay in [1, 2, 3]:
        p = StrategyParams(**{**params.__dict__, "entry_delay_bars": delay})
        trades = generate_trades(df, p, bar_minutes)
        pnl = sum(t.pnl for t in trades)
        deg = (baseline_pnl - pnl) / abs(baseline_pnl) if baseline_pnl != 0 else 0
        results[f"delay_{delay}"] = {
            "pnl": pnl,
            "degradation": deg,
            "n_trades": len(trades),
        }
    
    return results


def test_walk_forward(
    df: pd.DataFrame,
    params: StrategyParams,
    bar_minutes: int = 1,
    n_folds: int = 5,
    is_ratio: float = 0.7,
) -> dict:
    """
    Walk-forward analysis: split data into folds, train on IS, test on OOS.
    Uses the daily PnL from each fold period.
    """
    from strategy.vwap_strategy import generate_trades
    
    # Get unique dates
    dates = sorted(set(df.index.date))
    total_days = len(dates)
    
    if total_days < 30:
        return {"error": "Not enough data for walk-forward"}
    
    fold_size = total_days // n_folds
    is_days = int(fold_size * is_ratio)
    oos_days = fold_size - is_days
    
    is_pnls = []
    oos_pnls = []
    oos_win_rates = []
    
    for fold in range(n_folds):
        start = fold * fold_size
        end = min(start + fold_size, total_days)
        
        fold_dates = dates[start:end]
        if len(fold_dates) < 10:
            continue
        
        # Split into IS/OOS
        is_dates = set(fold_dates[:is_days])
        oos_dates = set(fold_dates[is_days:])
        
        # Generate trades for this fold
        fold_df = df[df.index.date.isin(set(fold_dates))]
        trades = generate_trades(fold_df, params, bar_minutes)
        
        # Split trades by date
        is_trades = [t for t in trades if t.entry_time.date() in is_dates]
        oos_trades = [t for t in trades if t.entry_time.date() in oos_dates]
        
        is_pnl = sum(t.pnl for t in is_trades)
        oos_pnl = sum(t.pnl for t in oos_trades)
        oos_wr = sum(1 for t in oos_trades if t.pnl > 0) / len(oos_trades) if oos_trades else 0
        
        is_pnls.append(is_pnl)
        oos_pnls.append(oos_pnl)
        oos_win_rates.append(oos_wr)
    
    avg_is = np.mean(is_pnls) if is_pnls else 0
    avg_oos = np.mean(oos_pnls) if oos_pnls else 0
    avg_oos_wr = np.mean(oos_win_rates) if oos_win_rates else 0
    efficiency = avg_oos / avg_is if avg_is > 0 else 0
    
    return {
        "n_folds": len(is_pnls),
        "is_pnl": avg_is,
        "oos_pnl": avg_oos,
        "oos_win_rate": avg_oos_wr,
        "efficiency": efficiency,
        "fold_is_pnls": is_pnls,
        "fold_oos_pnls": oos_pnls,
    }


def test_perturbation(
    df: pd.DataFrame,
    params: StrategyParams,
    bar_minutes: int = 1,
    n_perturbations: int = 10,
    noise_ticks: float = 1.0,
) -> dict:
    """
    Test sensitivity to price perturbation.
    Adds random noise to entry/exit prices and re-runs.
    """
    from strategy.vwap_strategy import generate_trades, Trade, TICK_SIZE, MNQ_TICK_VALUE
    
    # Baseline
    baseline = generate_trades(df, params, bar_minutes)
    baseline_pnl = sum(t.pnl for t in baseline)
    
    perturbed_pnls = []
    seed = 42
    rng = np.random.RandomState(seed)
    
    for _ in range(n_perturbations):
        # Add noise to each trade's entry and exit
        noise_entry = rng.normal(0, noise_ticks * TICK_SIZE, len(baseline))
        noise_exit = rng.normal(0, noise_ticks * TICK_SIZE, len(baseline))
        
        total = 0
        for i, t in enumerate(baseline):
            adj_entry = t.entry_price + noise_entry[i]
            adj_exit = t.exit_price + noise_exit[i]
            
            pnl_ticks = (adj_exit - adj_entry) / TICK_SIZE
            if t.side == "short":
                pnl_ticks = -pnl_ticks
            pnl = pnl_ticks * MNQ_TICK_VALUE * t.quantity - params.commission_per_side * 2 * t.quantity
            total += pnl
        
        perturbed_pnls.append(total)
    
    avg_perturbed = np.mean(perturbed_pnls)
    degradation = (baseline_pnl - avg_perturbed) / abs(baseline_pnl) if baseline_pnl != 0 else 0
    
    return {
        "baseline_pnl": baseline_pnl,
        "avg_perturbed_pnl": avg_perturbed,
        "degradation": degradation,
        "std_perturbed": np.std(perturbed_pnls),
        "min_perturbed": np.min(perturbed_pnls),
        "max_perturbed": np.max(perturbed_pnls),
    }


def test_multi_timeframe(
    df: pd.DataFrame,
    params: StrategyParams,
) -> dict:
    """Test same strategy across 1m, 5m, 15m bars."""
    results = {}
    for bar_min in [1, 5, 15]:
        trades = generate_trades(df, params, bar_min)
        pnl = sum(t.pnl for t in trades)
        wr = sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0
        results[f"{bar_min}m"] = {
            "pnl": pnl,
            "n_trades": len(trades),
            "win_rate": wr,
        }
    
    profitable_count = sum(1 for v in results.values() if v["pnl"] > 0)
    results["consistency"] = profitable_count / len(results)
    
    return results


def run_robustness_suite(
    nq_csv_path: str | Path,
    params: StrategyParams | None = None,
    bar_minutes: int = 5,
) -> RobustnessResult:
    """
    Run full robustness test suite.
    Returns RobustnessResult with pass/fail verdict.
    """
    if params is None:
        params = StrategyParams()
    
    df = load_nq_data(nq_csv_path)
    
    issues = []
    
    # 1. Baseline
    baseline_trades = generate_trades(df, params, bar_minutes)
    baseline_pnl = sum(t.pnl for t in baseline_trades)
    baseline_wr = sum(1 for t in baseline_trades if t.pnl > 0) / len(baseline_trades) if baseline_trades else 0
    
    # 2. Entry delay
    delay_results = test_entry_delay(df, params, bar_minutes)
    d1 = delay_results.get("delay_1", {"pnl": 0, "degradation": 1})
    d2 = delay_results.get("delay_2", {"pnl": 0, "degradation": 1})
    
    if d1["degradation"] > 0.30:
        issues.append(f"Entry delay 1: {d1['degradation']:.0%} PnL degradation (>30%)")
    if d2["degradation"] > 0.50:
        issues.append(f"Entry delay 2: {d2['degradation']:.0%} PnL degradation (>50%)")
    
    # 3. Walk-forward
    wf = test_walk_forward(df, params, bar_minutes)
    wf_eff = wf.get("efficiency", 0)
    wf_oos = wf.get("oos_pnl", 0)
    
    if wf_eff < 0.3:
        issues.append(f"Walk-forward efficiency: {wf_eff:.2f} (< 0.30)")
    if wf_oos < 0:
        issues.append(f"Walk-forward OOS PnL negative: ${wf_oos:,.0f}")
    
    # 4. Perturbation
    perturb = test_perturbation(df, params, bar_minutes)
    p_deg = perturb.get("degradation", 1)
    
    if p_deg > 0.30:
        issues.append(f"Perturbation degradation: {p_deg:.0%} (> 30%)")
    
    # 5. Multi-timeframe
    tf = test_multi_timeframe(df, params)
    tf_consistency = tf.get("consistency", 0)
    
    if tf_consistency < 0.5:
        issues.append(f"Multi-TF consistency: {tf_consistency:.0%} (< 50%)")
    
    # Verdict
    robust = len(issues) == 0
    
    return RobustnessResult(
        baseline_pnl=baseline_pnl,
        baseline_win_rate=baseline_wr,
        baseline_trades=len(baseline_trades),
        delay_1_pnl=d1["pnl"],
        delay_1_degradation=d1["degradation"],
        delay_2_pnl=d2["pnl"],
        delay_2_degradation=d2["degradation"],
        wf_oos_pnl=wf_oos,
        wf_oos_win_rate=wf.get("oos_win_rate", 0),
        wf_is_pnl=wf.get("is_pnl", 0),
        wf_efficiency=wf_eff,
        perturb_1tick_pnl=perturb.get("avg_perturbed_pnl", 0),
        perturb_1tick_degradation=p_deg,
        tf_5m_pnl=tf.get("5m", {}).get("pnl", 0),
        tf_15m_pnl=tf.get("15m", {}).get("pnl", 0),
        tf_consistency=tf_consistency,
        robust=robust,
        issues=issues,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    nq_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "nq-1m.csv"
    
    params = StrategyParams()
    bar_min = 5
    
    if len(sys.argv) > 1:
        params.entry_delay_bars = int(sys.argv[1])
    if len(sys.argv) > 2:
        bar_min = int(sys.argv[2])
    
    result = run_robustness_suite(nq_path, params, bar_min)
    
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS REPORT")
    print(f"{'='*60}")
    print(f"Baseline PnL:      ${result.baseline_pnl:>12,.2f}  ({result.baseline_trades} trades, {result.baseline_win_rate:.1%} WR)")
    print(f"Delay-1 PnL:       ${result.delay_1_pnl:>12,.2f}  (degradation: {result.delay_1_degradation:.1%})")
    print(f"Delay-2 PnL:       ${result.delay_2_pnl:>12,.2f}  (degradation: {result.delay_2_degradation:.1%})")
    print(f"WF OOS PnL:        ${result.wf_oos_pnl:>12,.2f}  (efficiency: {result.wf_efficiency:.2f})")
    print(f"Perturb PnL:       ${result.perturb_1tick_pnl:>12,.2f}  (degradation: {result.perturb_1tick_degradation:.1%})")
    print(f"5m TF PnL:         ${result.tf_5m_pnl:>12,.2f}")
    print(f"15m TF PnL:        ${result.tf_15m_pnl:>12,.2f}")
    print(f"Multi-TF consistency: {result.tf_consistency:.0%}")
    print(f"{'='*60}")
    if result.robust:
        print(f"✓ ROBUST — passes all checks")
    else:
        print(f"✗ NOT ROBUST — {len(result.issues)} issue(s):")
        for issue in result.issues:
            print(f"  - {issue}")
