"""
benchmark.py — Autoresearch harness for Prop-Firm EV Calculator.

Runs the full TraderLaunch evaluation pipeline with deterministic Monte Carlo
seeds and reports key metrics for optimization tracking.

Primary metric: ev_per_pipeline_usd (Expected Value per full pipeline)
Secondary metrics: challenge_pass_rate, funded_pass_rate, live_trader_profit,
                   prob_reaching_live, live_total_withdrawn, live_days_traded
"""

import sys
import time
import logging
from pathlib import Path

# Suppress all logging to keep benchmark output clean
logging.disable(logging.CRITICAL)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from Prop_firm.Trader_launch.Trader_launch_calculator import (
    load_trades,
    ensure_directories,
    TRADES_CSV,
    RESULTS_DIR,
    FIGURES_DIR,
)
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Challenge_phase import (
    run as run_challenge_rules_filter,
)
from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import (
    run as run_challenge_phase,
)
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Funded_phase import (
    run as run_funded_rules_filter,
)
from Prop_firm.Trader_launch.calculator_logic.Funded_phase import (
    run as run_funded_phase,
)
from Prop_firm.Trader_launch.calculator_logic.Live_phase import (
    run as run_live_phase,
)
from Prop_firm.Trader_launch.calculator_logic.Final_result import (
    run as run_final_result,
)

# Deterministic MC seed
MC_SEED = 42
MC_CHALLENGE_BLOCK_SIZE = 28
MC_CHALLENGE_CIRCULAR = True
MC_FUNDED_BLOCK_SIZE = 6


# ── Hybrid MC helper ───────────────────────────────────────────────────
def _apply_lock_numpy(pnl, hup, lock_amount):
    """Fast lock rule on arrays. Returns total PnL for the day after lock."""
    running = 0.0
    for i in range(len(pnl)):
        if running + hup[i] >= lock_amount:
            room = lock_amount - running
            return running + min(pnl[i], room)
        running += pnl[i]
    return running


def _setup_trade_arrays(trades, profit_target, consistency_pct):
    """Pre-extract trade arrays by day for fast MC."""
    lock_amount = profit_target * consistency_pct / 100.0
    day_labels = trades.index.to_series().dt.normalize()
    day_arrays = []
    for day in sorted(day_labels.unique()):
        indices = day_labels[day_labels == day].index.tolist()
        pnl = trades.loc[indices, 'pnl'].values.copy()
        hup = trades.loc[indices, 'highest_unrealized_profit'].values.copy()
        day_arrays.append((pnl, hup))
    return day_arrays, lock_amount


def _generate_daily_pnl(rng, day_arrays, lock_amount):
    """Generate one daily PnL sequence via trade-level shuffle + lock."""
    n_days = len(day_arrays)
    daily = np.empty(n_days)
    for i, (pnl, hup) in enumerate(day_arrays):
        n = len(pnl)
        if n > 1:
            perm = rng.permutation(n)
            daily[i] = _apply_lock_numpy(pnl[perm], hup[perm], lock_amount)
        else:
            daily[i] = _apply_lock_numpy(pnl, hup, lock_amount)
    return daily


_HYBRID_DAY_ARRAYS = None
_HYBRID_LOCK_AMOUNT = None


def _hybrid_run_simulations(daily_pnl, start_balance, max_drawdown, profit_target,
                            n_simulations=10000, max_daily_loss=None, max_trading_days=None,
                            min_trading_days=1, seed=None, keep_details=False,
                            block_size=1, circular=False):
    """Hybrid MC: trade-level lock variation + block bootstrap on daily PnL."""
    from Common.EOD.trailing_drawdown import simulate_pnl_sequence
    from Common.Monte_carlo.simulator import SimulationResult
    
    rng = np.random.default_rng(seed or 42)
    n_inner = 5
    n_outer = n_simulations // n_inner
    
    pass_days = []
    fail_days = []
    max_dds = []
    final_pnls = []
    total_pass = 0
    total_fail = 0
    total_runs = 0
    
    for outer in range(n_outer):
        dpnl = _generate_daily_pnl(rng, _HYBRID_DAY_ARRAYS, _HYBRID_LOCK_AMOUNT)
        n = len(dpnl)
        
        for inner in range(n_inner):
            blocks = []
            pos = 0
            while pos < n:
                start = rng.integers(0, n)
                block = dpnl[(np.arange(start, start + block_size) % n)]
                blocks.append(block)
                pos += block_size
            shuffled = np.concatenate(blocks)[:n]
            
            result = simulate_pnl_sequence(
                pnl_values=shuffled, start_balance=start_balance,
                max_drawdown=max_drawdown, profit_target=profit_target,
                min_trading_days=min_trading_days,
            )
            max_dds.append(result.max_trailing_dd)
            final_pnls.append(result.total_pnl)
            if result.passed:
                total_pass += 1
                pass_days.append(result.day_profit_target_reached or 0)
            elif result.breached:
                total_fail += 1
                fail_days.append(result.day_breached or 0)
            total_runs += 1
    
    pass_rate = 100.0 * total_pass / total_runs
    fail_rate = 100.0 * total_fail / total_runs
    
    def _stats(days):
        if not days:
            return None, None, None
        return float(np.mean(days)), float(np.std(days)), float(np.median(days))
    
    avg_pass, std_pass, med_pass = _stats(pass_days)
    avg_fail, std_fail, med_fail = _stats(fail_days)
    
    return SimulationResult(
        n_simulations=total_runs, pass_rate=pass_rate, fail_rate=fail_rate,
        time_expired_rate=100.0 * (total_runs - total_pass - total_fail) / total_runs,
        avg_pass_days=avg_pass, std_pass_days=std_pass, median_pass_days=med_pass,
        avg_fail_days=avg_fail, std_fail_days=std_fail, median_fail_days=med_fail,
        avg_max_drawdown=float(np.mean(max_dds)),
        avg_final_pnl=float(np.mean(final_pnls)),
        run_details=[],
    )


def main() -> int:
    """Run the full pipeline with deterministic seeds and emit METRIC lines."""
    ensure_directories()
    trades = load_trades(TRADES_CSV)

    t0 = time.perf_counter()

    # Step 1: Challenge rules filter
    challenge_rules_result = run_challenge_rules_filter(trades=trades, config=None)

    # Step 2: Challenge phase with hybrid MC (trade-level lock + block bootstrap)
    import Common.Monte_carlo.simulator as _sim_mod
    _orig_run_sim = _sim_mod.run_simulations
    
    # Setup trade arrays for hybrid MC
    import Prop_firm.Trader_launch.calculator_logic.Challenge_phase as _ch_mod
    _orig_run_sim_ch = _ch_mod.run_simulations
    from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import _load_rules as _lr
    _rules = _lr()
    _cfg = _rules["phases"]["challenge"]
    global _HYBRID_DAY_ARRAYS, _HYBRID_LOCK_AMOUNT
    _HYBRID_DAY_ARRAYS, _HYBRID_LOCK_AMOUNT = _setup_trade_arrays(
        trades, float(_cfg["profit_target"]), float(_cfg["consistency_rule"]),
    )
    _ch_mod.run_simulations = _hybrid_run_simulations
    
    challenge_result = run_challenge_phase(
        trades=trades,
        config=None,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        mc_seed=MC_SEED,
        mc_block_size=MC_CHALLENGE_BLOCK_SIZE,
        mc_circular=MC_CHALLENGE_CIRCULAR,
    )
    
    # Restore original for funded phase
    _ch_mod.run_simulations = _orig_run_sim_ch

    # Step 3: Funded rules filter
    funded_rules_result = run_funded_rules_filter(trades=trades, config=None)

    # Step 4: Funded phase (with deterministic MC seed)
    funded_result = run_funded_phase(
        trades=trades,
        challenge_result=challenge_result,
        config=None,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        mc_seed=MC_SEED,
        mc_block_size=MC_FUNDED_BLOCK_SIZE,
        mc_circular=MC_CHALLENGE_CIRCULAR,
    )

    # Step 5: Live phase
    live_result = run_live_phase(
        trades=trades,
        funded_result=funded_result,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # Step 6: Final result
    final_result = run_final_result(
        challenge_result=challenge_result,
        funded_result=funded_result,
        live_result=live_result,
        reports_dir=RESULTS_DIR,
    )

    elapsed = time.perf_counter() - t0

    # Read the final result metadata (written by step 6)
    metadata_path = RESULTS_DIR / "Final_result_metadata.yaml"
    if not metadata_path.exists():
        print("ERROR: Final_result_metadata.yaml not found", file=sys.stderr)
        return 1

    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)

    # Extract metrics
    challenge_pass_rate = meta.get("challenge", {}).get("mc_pass_rate_pct", 0.0)
    funded_pass_rate = meta.get("funded", {}).get("mc_pass_rate_pct", 0.0)
    prob_reaching_live = meta.get("final", {}).get("prob_reaching_live_pct", 0.0)
    ev_per_pipeline = meta.get("final", {}).get("expected_value_per_pipeline_usd", 0.0)
    ev_per_attempt = meta.get("final", {}).get("expected_value_per_attempt_usd", 0.0)
    live_trader_profit = meta.get("live", {}).get("trader_profit_usd", 0.0)
    live_total_withdrawn = meta.get("live", {}).get("total_withdrawn_usd", 0.0)

    # Live phase details
    live_summary = live_result.get("summary", {}) if isinstance(live_result, dict) else {}
    live_days = live_summary.get("days_traded", 0)

    # Emit METRIC lines (primary first)
    print(f"METRIC ev_per_pipeline_usd={ev_per_pipeline:.2f}")
    print(f"METRIC challenge_pass_rate={challenge_pass_rate:.2f}")
    print(f"METRIC funded_pass_rate={funded_pass_rate:.2f}")
    print(f"METRIC prob_reaching_live={prob_reaching_live:.2f}")
    print(f"METRIC ev_per_attempt_usd={ev_per_attempt:.2f}")
    print(f"METRIC live_trader_profit_usd={live_trader_profit:.2f}")
    print(f"METRIC live_total_withdrawn_usd={live_total_withdrawn:.2f}")
    print(f"METRIC live_days_traded={live_days}")
    print(f"METRIC pipeline_runtime_s={elapsed:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
