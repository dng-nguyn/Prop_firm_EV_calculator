"""
Sequential Monte Carlo Pipeline for Prop Firm EV
=================================================
For each MC trial:
1. Shuffle daily PnL once
2. Challenge phase: simulate with $2k target, $1k DD, min 3 days, max 60 days, 40% consistency
3. If challenge passes: slice remaining PnL, run funded phase ($1k target, $1k DD)
4. If funded passes: simulate live phase ($101k start, $100k termination, 55% split)

This is the correct pipeline per the advisory — NOT independent simulations on same data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from Common.EOD.trailing_drawdown import EODResult, simulate_pnl_sequence

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of one sequential MC trial."""
    challenge_passed: bool
    challenge_days: int
    challenge_pnl: float
    challenge_breached: bool
    
    funded_passed: bool
    funded_days: int
    funded_pnl: float
    
    live_profit: float
    live_days: int
    live_terminated: bool
    
    total_trader_profit: float  # After profit split


@dataclass
class SequentialSimResult:
    """Aggregated results from sequential MC simulations."""
    n_simulations: int
    
    # Challenge stats
    challenge_pass_rate: float
    challenge_avg_days: Optional[float]
    
    # Pipeline stats (challenge → funded → live)
    pipeline_pass_rate: float
    funded_pass_rate: float  # Of those who passed challenge
    
    # EV
    avg_live_profit: float
    avg_trader_profit: float
    ev_per_pipeline: float
    ev_per_attempt: float
    
    # Per-trial details
    results: list[PipelineResult]


def apply_consistency_clip(
    daily_pnl: np.ndarray,
    consistency_pct: float = 0.40,
) -> np.ndarray:
    """
    Enforce 40% consistency rule: clip each day's PnL so no single day
    exceeds consistency_pct of running gross profits.
    
    Per rules: "auto close your trades and lock you out for the day"
    when 40% of overall profits is reached.
    
    This means: as cumulative gross profit grows, the daily cap grows.
    Early days have tighter caps.
    """
    clipped = daily_pnl.copy()
    running_gross = 0.0
    
    for i in range(len(clipped)):
        if clipped[i] > 0:
            # Daily cap = consistency_pct * running gross profits
            # But "overall profits" means total gross profits so far INCLUDING this day
            # So cap = consistency_pct * (running_gross + this_day_profit)
            # Solving: day_profit <= consistency_pct * (running_gross + day_profit)
            # day_profit * (1 - consistency_pct) <= consistency_pct * running_gross
            # day_profit <= (consistency_pct / (1 - consistency_pct)) * running_gross
            if running_gross > 0:
                cap = (consistency_pct / (1.0 - consistency_pct)) * running_gross
                clipped[i] = min(clipped[i], cap)
            running_gross += max(0, clipped[i])
    
    return clipped


def simulate_live_phase(
    daily_pnl: np.ndarray,
    start_balance: float = 101_000.0,
    termination: float = 100_000.0,
    withdrawal_threshold: float = 101_200.0,
    buffer: float = 101_000.0,
    profit_split: float = 0.55,
) -> tuple[float, int, bool]:
    """
    Simulate live funded phase with profit withdrawals.
    
    Returns: (total_trader_profit, days_traded, terminated)
    """
    balance = start_balance
    total_trader_profit = 0.0
    days = 0
    terminated = False
    
    for pnl in daily_pnl:
        balance += pnl
        days += 1
        
        if balance <= termination:
            terminated = True
            break
        
        # Withdraw profit above threshold
        if balance >= withdrawal_threshold:
            profit = balance - buffer
            if profit >= 200:  # Min payout $200
                trader_share = profit * profit_split
                total_trader_profit += trader_share
                balance = buffer
    
    return total_trader_profit, days, terminated


def run_sequential_simulation(
    daily_pnl: np.ndarray,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
    # Challenge rules (from rule_sets_metadata.yaml)
    chal_start: float = 100_000.0,
    chal_dd: float = 1_000.0,
    chal_target: float = 2_000.0,
    chal_min_days: int = 3,
    chal_max_days: int = 60,
    chal_consistency: float = 0.40,
    # Funded rules
    fund_dd: float = 1_000.0,
    fund_target: float = 1_000.0,
    # Live rules (from rule_sets_payout.txt)
    live_start: float = 101_000.0,
    live_termination: float = 100_000.0,
    live_withdrawal: float = 101_200.0,
    live_buffer: float = 101_000.0,
    profit_split: float = 0.55,
    eval_fee: float = 45.0,
) -> SequentialSimResult:
    """
    Run N sequential MC trials: Challenge → Funded → Live.
    
    Each trial:
    1. Shuffle daily PnL
    2. Apply 40% consistency clip for challenge phase
    3. Run challenge simulation
    4. If passed, slice remaining PnL for funded
    5. If funded passed, simulate live on remaining PnL
    """
    rng = np.random.default_rng(seed)
    
    results: list[PipelineResult] = []
    chal_pass_count = 0
    pipeline_pass_count = 0
    funded_pass_count = 0
    live_profits: list[float] = []
    trader_profits: list[float] = []
    
    for _ in range(n_simulations):
        # 1. Shuffle
        shuffled = daily_pnl.copy()
        rng.shuffle(shuffled)
        
        # 2. Apply consistency clip for challenge
        chal_pnl = apply_consistency_clip(shuffled, chal_consistency)
        
        # 3. Challenge phase
        chal_result = simulate_pnl_sequence(
            pnl_values=chal_pnl,
            start_balance=chal_start,
            max_drawdown=chal_dd,
            profit_target=chal_target,
            min_trading_days=chal_min_days,
            max_trading_days=chal_max_days,
        )
        
        pr = PipelineResult(
            challenge_passed=chal_result.passed,
            challenge_days=chal_result.days_traded,
            challenge_pnl=chal_result.total_pnl,
            challenge_breached=chal_result.breached,
            funded_passed=False,
            funded_days=0,
            funded_pnl=0.0,
            live_profit=0.0,
            live_days=0,
            live_terminated=False,
            total_trader_profit=0.0,
        )
        
        if not chal_result.passed:
            results.append(pr)
            continue
        
        chal_pass_count += 1
        
        # 4. Slice remaining PnL for funded (after challenge pass day)
        pass_day = chal_result.day_profit_target_reached  # 1-based
        remaining = shuffled[pass_day:]  # PnL after challenge pass
        
        if len(remaining) == 0:
            results.append(pr)
            continue
        
        # Funded phase: NO consistency rule, NO min days
        fund_result = simulate_pnl_sequence(
            pnl_values=remaining,
            start_balance=chal_start,  # Same account balance
            max_drawdown=fund_dd,
            profit_target=fund_target,
            min_trading_days=1,  # No min days per rules
        )
        
        pr.funded_passed = fund_result.passed
        pr.funded_days = fund_result.days_traded
        pr.funded_pnl = fund_result.total_pnl
        
        if not fund_result.passed:
            results.append(pr)
            continue
        
        funded_pass_count += 1
        
        # 5. Live phase: slice remaining PnL after funded pass
        fund_pass_day = fund_result.day_profit_target_reached
        live_remaining = remaining[fund_pass_day:]
        
        trader_profit, live_days, terminated = simulate_live_phase(
            live_remaining,
            start_balance=live_start,
            termination=live_termination,
            withdrawal_threshold=live_withdrawal,
            buffer=live_buffer,
            profit_split=profit_split,
        )
        
        pr.live_profit = trader_profit / profit_split if profit_split > 0 else 0  # Gross
        pr.live_days = live_days
        pr.live_terminated = terminated
        pr.total_trader_profit = trader_profit
        
        if trader_profit > 0:
            pipeline_pass_count += 1
        
        live_profits.append(pr.live_profit)
        trader_profits.append(trader_profit)
        
        results.append(pr)
    
    # Aggregate
    chal_pass_rate = 100.0 * chal_pass_count / n_simulations if n_simulations > 0 else 0
    funded_pass_rate = 100.0 * funded_pass_count / chal_pass_count if chal_pass_count > 0 else 0
    pipeline_pass_rate = 100.0 * pipeline_pass_count / n_simulations if n_simulations > 0 else 0
    
    avg_live = np.mean(live_profits) if live_profits else 0
    avg_trader = np.mean(trader_profits) if trader_profits else 0
    
    # EV = (pipeline_pass_rate * avg_trader_profit) - eval_fee
    pipeline_prob = pipeline_pass_count / n_simulations if n_simulations > 0 else 0
    ev_per_pipeline = pipeline_prob * avg_trader - eval_fee
    
    chal_prob = chal_pass_count / n_simulations if n_simulations > 0 else 0
    ev_per_attempt = chal_prob * (funded_pass_count / chal_pass_count if chal_pass_count > 0 else 0) * avg_trader - eval_fee if chal_pass_count > 0 else -eval_fee
    
    # Challenge avg days
    chal_days_list = [r.challenge_days for r in results if r.challenge_passed]
    chal_avg_days = np.mean(chal_days_list) if chal_days_list else None
    
    return SequentialSimResult(
        n_simulations=n_simulations,
        challenge_pass_rate=chal_pass_rate,
        challenge_avg_days=chal_avg_days,
        pipeline_pass_rate=pipeline_pass_rate,
        funded_pass_rate=funded_pass_rate,
        avg_live_profit=avg_live,
        avg_trader_profit=avg_trader,
        ev_per_pipeline=ev_per_pipeline,
        ev_per_attempt=ev_per_attempt,
        results=results,
    )
