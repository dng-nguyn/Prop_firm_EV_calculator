"""
Monte Carlo — Random‑shuffle Simulation Engine
===============================================

Takes the observed daily PnL sequence from a trader's history, shuffles it
many times, and for each shuffled sequence evaluates whether the prop‑firm
rules would have been met (profit target reached before max drawdown
breach).  The fraction of successful simulations is an estimate of the
probability of passing the challenge/funded phase.

Key improvement over naive approach
-----------------------------------
Uses the low-level ``simulate_pnl_sequence`` from the EOD module, which
accepts a raw numpy array of daily PnL values — no fake DataFrames needed.
This is significantly faster for large batch simulations.

Typical usage::

    from Common.EOD import aggregate_daily_pnl
    from Common.Monte_carlo import run_simulations

    daily_pnl = aggregate_daily_pnl(trades)
    result = run_simulations(
        daily_pnl=daily_pnl,
        start_balance=100_000.0,
        max_drawdown=1_000.0,
        profit_target=2_000.0,
        n_simulations=10_000,
        seed=42,
    )
    print(f"Pass rate: {result.pass_rate:.1f}%")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from Common.EOD.trailing_drawdown import EODResult, simulate_pnl_sequence

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Aggregated statistics across all Monte Carlo runs."""

    n_simulations: int
    pass_rate: float               # % of simulations that hit profit target
    fail_rate: float               # % that breached drawdown
    time_expired_rate: float       # % that ran out of time

    avg_pass_days: Optional[float]        # Mean days to profit target (pass only)
    std_pass_days: Optional[float]
    median_pass_days: Optional[float]

    avg_fail_days: Optional[float]        # Mean days to drawdown breach (fail only)
    std_fail_days: Optional[float]
    median_fail_days: Optional[float]

    avg_max_drawdown: float               # Mean max trailing drawdown across all runs
    avg_final_pnl: float                  # Mean total PnL across all runs

    # Detailed breakdown per run (optional, useful for plotting)
    run_details: list[EODResult] = field(repr=False, default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# Core simulation logic
# ──────────────────────────────────────────────────────────────────────────



def _auto_block_size(daily_pnl: pd.Series, block_fraction: float = 0.0) -> int:
    """Automatically select block size based on data length.
    
    Default: n//4 (25% of data). Configurable via block_fraction parameter.
    - 30 days → 7
    - 42 days → 10
    - 60 days → 15
    - 120 days → 30
    
    Args:
        daily_pnl: Daily PnL series
        block_fraction: Override fraction (0.0 = use default n//4)
    """
    n = len(daily_pnl)
    if n < 10:
        return 1
    if block_fraction > 0:
        return max(5, min(int(n * block_fraction), n // 2))
    return max(5, min(n // 4, n // 2))


def run_simulations(
    daily_pnl: pd.Series,
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    n_simulations: int = 10_000,
    max_daily_loss: Optional[float] = None,
    max_trading_days: Optional[int] = None,
    min_trading_days: int = 1,
    seed: Optional[int] = None,
    keep_details: bool = False,
    block_size: int = 0,  # 0 = auto-select based on data
    circular: bool = False,
    floor_aware: bool = False,
) -> SimulationResult:
    """
    Run *n* Monte‑Carlo simulations by randomly shuffling the observed
    daily PnL sequence.

    Each simulation evaluates the shuffled sequence against the same
    EOD trailing‑drawdown rules.  The pass rate is the fraction of
    simulations that hit the profit target before breaching the max
    drawdown.

    Parameters
    ----------
    daily_pnl : pd.Series
        Observed daily PnL values (index by date, values are net PnL).
        The values are shuffled — the index is discarded during simulation.
    start_balance : float
        Initial account balance.
    max_drawdown : float
        Maximum allowed trailing drawdown (dollars).
    profit_target : float
        Net profit needed to pass the phase.
    n_simulations : int
        Number of random shuffles to run.
    max_daily_loss : float or None, optional
        Maximum allowed single‑day loss.
    max_trading_days : int or None, optional
        Hard cap on trading days per simulation.
    min_trading_days : int
        Minimum trading days before profit target is accepted.
    seed : int or None, optional
        Random seed for reproducibility.
    keep_details : bool
        If True, store every EODResult in ``run_details`` (memory heavy
        for large ``n_simulations``).
    block_size : int
        Size of blocks for block-bootstrap resampling. 1 = pure random
        shuffle (default). Values > 1 preserve short-term autocorrelation
        in the daily PnL sequence. Recommended: 5-10 for typical trading data.

    Returns
    -------
    SimulationResult
    """
    if len(daily_pnl) == 0:
        return SimulationResult(
            n_simulations=n_simulations,
            pass_rate=0.0,
            fail_rate=0.0,
            time_expired_rate=0.0,
            avg_pass_days=None,
            std_pass_days=None,
            median_pass_days=None,
            avg_fail_days=None,
            std_fail_days=None,
            median_fail_days=None,
            avg_max_drawdown=0.0,
            avg_final_pnl=0.0,
        )

    # Auto-select block size if not specified
    if block_size <= 0:
        block_size = _auto_block_size(daily_pnl)
    
    rng = np.random.default_rng(seed)
    pnl_values = daily_pnl.values  # raw numpy array for speed

    results: list[EODResult] = []
    pass_days: list[int] = []
    fail_days: list[int] = []
    max_dds: list[float] = []
    final_pnls: list[float] = []

    n = len(pnl_values)
    for _ in range(n_simulations):
        if block_size <= 1:
            # Pure random shuffle
            shuffled = pnl_values.copy()
            rng.shuffle(shuffled)
        else:
            # Block bootstrap: sample contiguous blocks with replacement
            blocks = []
            pos = 0
            while pos < n:
                if circular:
                    # Circular: wrap around to avoid edge effects
                    start = rng.integers(0, n)
                    block = np.array([pnl_values[(start + j) % n] for j in range(block_size)])
                else:
                    # Non-circular: only sample blocks that fit within bounds
                    start = rng.integers(0, max(1, n - block_size + 1))
                    block = pnl_values[start:start + block_size]
                blocks.append(block)
                pos += block_size
            shuffled = np.concatenate(blocks)[:n]

        # Call low-level EOD function directly — no fake DataFrame needed
        result = simulate_pnl_sequence(
            pnl_values=shuffled,
            start_balance=start_balance,
            max_drawdown=max_drawdown,
            profit_target=profit_target,
            max_daily_loss=max_daily_loss,
            max_trading_days=max_trading_days,
            min_trading_days=min_trading_days,
            floor_aware=floor_aware,
        )

        results.append(result)
        max_dds.append(result.max_trailing_dd)
        final_pnls.append(result.total_pnl)

        if result.passed:
            pass_days.append(result.day_profit_target_reached or 0)
        elif result.breached:
            fail_days.append(result.day_breached or 0)

    # ---- Aggregate -------------------------------------------------------
    n_pass = len(pass_days)
    n_fail = len(fail_days)
    n_time = n_simulations - n_pass - n_fail

    pass_rate = 100.0 * n_pass / n_simulations
    fail_rate = 100.0 * n_fail / n_simulations
    time_expired_rate = 100.0 * n_time / n_simulations

    def _stats(days: list[int]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not days:
            return None, None, None
        return float(np.mean(days)), float(np.std(days)), float(np.median(days))

    avg_pass, std_pass, med_pass = _stats(pass_days)
    avg_fail, std_fail, med_fail = _stats(fail_days)

    return SimulationResult(
        n_simulations=n_simulations,
        pass_rate=pass_rate,
        fail_rate=fail_rate,
        time_expired_rate=time_expired_rate,
        avg_pass_days=avg_pass,
        std_pass_days=std_pass,
        median_pass_days=med_pass,
        avg_fail_days=avg_fail,
        std_fail_days=std_fail,
        median_fail_days=med_fail,
        avg_max_drawdown=float(np.mean(max_dds)),
        avg_final_pnl=float(np.mean(final_pnls)),
        run_details=results if keep_details else [],
    )


def run_full_analysis(
    daily_pnl: pd.Series,
    eval_fee: float,
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    n_simulations: int = 10_000,
    max_daily_loss: Optional[float] = None,
    max_trading_days: Optional[int] = None,
    min_trading_days: int = 1,
    seed: Optional[int] = None,
    block_size: int = 1,
) -> dict:
    """
    Run Monte‑Carlo analysis and return a human‑friendly summary dict.

    In addition to the simulation pass/fail rates, this also computes:

    - **Expected cost**: average number of evaluation fees spent.
    - **Expected value**: (pass_rate × profit_target) − (fail_rate × fees).

    Parameters
    ----------
    daily_pnl : pd.Series
        Observed daily PnL values.
    eval_fee : float
        Cost of the evaluation (one‑time fee).
    start_balance : float
        Initial account balance.
    max_drawdown : float
        Maximum allowed trailing drawdown (dollars).
    profit_target : float
        Net profit needed to pass the phase.
    n_simulations : int
        Number of Monte Carlo runs.
    max_daily_loss : float or None, optional
        Maximum allowed single‑day loss.
    max_trading_days : int or None, optional
        Hard cap on trading days per simulation.
    min_trading_days : int
        Minimum trading days before profit target is accepted.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    dict
    """
    sim_result = run_simulations(
        daily_pnl=daily_pnl,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
        n_simulations=n_simulations,
        max_daily_loss=max_daily_loss,
        max_trading_days=max_trading_days,
        min_trading_days=min_trading_days,
        seed=seed,
        keep_details=False,
        block_size=block_size,
        circular=circular,
    )

    # Correct geometric series: expected_attempts = 1/pass_rate, so expected_cost = fee / pass_rate
    if sim_result.pass_rate > 0:
        expected_attempts = 100.0 / sim_result.pass_rate
        expected_cost = eval_fee * expected_attempts
        expected_value = profit_target - expected_cost
    else:
        expected_attempts = float('inf')
        expected_cost = float('inf')
        expected_value = float('-inf')

    return {
        "n_simulations": sim_result.n_simulations,
        "pass_rate_pct": round(sim_result.pass_rate, 2),
        "fail_rate_pct": round(sim_result.fail_rate, 2),
        "time_expired_rate_pct": round(sim_result.time_expired_rate, 2),
        "avg_pass_days": round(sim_result.avg_pass_days, 1) if sim_result.avg_pass_days is not None else None,
        "median_pass_days": sim_result.median_pass_days,
        "avg_fail_days": round(sim_result.avg_fail_days, 1) if sim_result.avg_fail_days is not None else None,
        "avg_max_drawdown": round(sim_result.avg_max_drawdown, 2),
        "avg_final_pnl": round(sim_result.avg_final_pnl, 2),
        "evaluation_fee": eval_fee,
        "expected_cost": round(expected_cost, 2),
        "expected_value": round(expected_value, 2),
    }
