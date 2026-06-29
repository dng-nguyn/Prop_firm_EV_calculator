"""
EOD — End-of-Day Trailing Drawdown Calculator

Core logic for tracking account equity, trailing drawdown, and
profit-target checks on a day-by-day basis.  Designed to be imported
by any prop‑firm calculator module.

Exports
-------
EODSnapshot, EODResult        — data‑class containers
aggregate_daily_pnl           — group trade PnL by calendar day
compute_equity_curve          — build equity curve from daily PnL
compute_trailing_drawdown     — compute trailing drawdown series
simulate_pnl_sequence         — low-level: walk through raw PnL values
simulate                      — mid-level: aggregate trades then simulate
analyze                       — high-level: convenience wrapper → dict
"""

from .trailing_drawdown import (
    EODResult,
    EODSnapshot,
    aggregate_daily_pnl,
    analyze,
    compute_equity_curve,
    compute_trailing_drawdown,
    simulate,
    simulate_pnl_sequence,
)

__all__ = [
    "EODResult",
    "EODSnapshot",
    "aggregate_daily_pnl",
    "analyze",
    "compute_equity_curve",
    "compute_trailing_drawdown",
    "simulate",
    "simulate_pnl_sequence",
]
