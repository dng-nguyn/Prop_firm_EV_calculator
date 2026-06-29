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


# ── Trade-level MC helper ───────────────────────────────────────────────
def _apply_lock_numpy(pnl, hup, lock_amount):
    """Fast lock rule on arrays. Returns total PnL for the day after lock."""
    running = 0.0
    for i in range(len(pnl)):
        if running + hup[i] >= lock_amount:
            room = lock_amount - running
            return running + min(pnl[i], room)
        running += pnl[i]
    return running


def _trade_level_mc(trades, n_sims, profit_target, consistency_pct, seed=42):
    """Trade-level MC: shuffle trades within each day, re-apply lock rule."""
    rng = np.random.default_rng(seed)
    lock_amount = profit_target * consistency_pct / 100.0
    day_labels = trades.index.to_series().dt.normalize()
    day_arrays = []
    for day in sorted(day_labels.unique()):
        indices = day_labels[day_labels == day].index.tolist()
        pnl = trades.loc[indices, 'pnl'].values.copy()
        hup = trades.loc[indices, 'highest_unrealized_profit'].values.copy()
        day_arrays.append((pnl, hup))
    n_days = len(day_arrays)
    pass_count = 0
    for _ in range(n_sims):
        daily_pnls = np.empty(n_days)
        for i, (pnl, hup) in enumerate(day_arrays):
            n = len(pnl)
            if n > 1:
                perm = rng.permutation(n)
                daily_pnls[i] = _apply_lock_numpy(pnl[perm], hup[perm], lock_amount)
            else:
                daily_pnls[i] = _apply_lock_numpy(pnl, hup, lock_amount)
        from Common.EOD.trailing_drawdown import simulate_pnl_sequence
        result = simulate_pnl_sequence(
            pnl_values=daily_pnls, start_balance=100_000,
            max_drawdown=1_000, profit_target=profit_target, min_trading_days=3,
        )
        if result.passed:
            pass_count += 1
    return 100.0 * pass_count / n_sims


def main() -> int:
    """Run the full pipeline with deterministic seeds and emit METRIC lines."""
    ensure_directories()
    trades = load_trades(TRADES_CSV)

    t0 = time.perf_counter()

    # Step 1: Challenge rules filter
    challenge_rules_result = run_challenge_rules_filter(trades=trades, config=None)

    # Step 2: Challenge phase (with deterministic MC seed)
    challenge_result = run_challenge_phase(
        trades=trades,
        config=None,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        mc_seed=MC_SEED,
        mc_block_size=MC_CHALLENGE_BLOCK_SIZE,
        mc_circular=MC_CHALLENGE_CIRCULAR,
    )

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

    # ── Trade-level MC challenge pass rate (conservative estimate) ──
    from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import _load_rules as _lr
    _rules = _lr()
    _cfg = _rules["phases"]["challenge"]
    challenge_pass_rate_tl = _trade_level_mc(
        trades, 5000,
        float(_cfg["profit_target"]),
        float(_cfg["consistency_rule"]),
        seed=MC_SEED,
    )
    eval_fee = float(_rules.get("general", {}).get("evaluation_fee", 45))
    funded_pass_rate_dec = funded_pass_rate / 100.0
    prob_live_tl = (challenge_pass_rate_tl / 100.0) * funded_pass_rate_dec
    ev_tl = live_trader_profit - eval_fee / prob_live_tl if prob_live_tl > 0 else 0.0

    # Emit METRIC lines (primary first)
    print(f"METRIC ev_per_pipeline_usd={ev_per_pipeline:.2f}")
    print(f"METRIC challenge_pass_rate={challenge_pass_rate:.2f}")
    print(f"METRIC challenge_pass_rate_tl={challenge_pass_rate_tl:.2f}")
    print(f"METRIC funded_pass_rate={funded_pass_rate:.2f}")
    print(f"METRIC prob_reaching_live={prob_reaching_live:.2f}")
    print(f"METRIC ev_per_attempt_usd={ev_per_attempt:.2f}")
    print(f"METRIC ev_per_pipeline_tl_usd={ev_tl:.2f}")
    print(f"METRIC live_trader_profit_usd={live_trader_profit:.2f}")
    print(f"METRIC live_total_withdrawn_usd={live_total_withdrawn:.2f}")
    print(f"METRIC live_days_traded={live_days}")
    print(f"METRIC pipeline_runtime_s={elapsed:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
