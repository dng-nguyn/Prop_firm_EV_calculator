"""
EOD — End-of-Day Trailing Drawdown Calculator
==============================================

Core logic for tracking account equity and an **EOD floor-based drawdown**
on a day-by-day basis.  Designed to be imported by any prop‑firm calculator
(Challenge or Funded phase).

How the EOD drawdown works (Trader Launch style)
-------------------------------------------------
1. **Floor** — a minimum balance that your account must stay above.
   - Initially: ``floor = start_balance - max_drawdown``.
   - The floor **trails up** when you make profits:
     ``floor = max(floor, peak_equity - max_drawdown)``.
   - The floor **never** moves down when you lose money.
2. **Floor lock** — once ``peak_equity >= start_balance + max_drawdown``,
   the floor **locks** at ``start_balance`` and stops trailing.
   This means your available drawdown room can grow beyond ``max_drawdown``.
3. **Breach** — you fail when ``equity < floor``.
4. **Trailing drawdown** (informational) — ``peak_equity - equity``.
   This is reported but is **not** the breach trigger.

Concepts
--------
*Starting balance* — the initial simulated account size (e.g. $100 000).
*Daily PnL* — sum of all trade PnLs for a single calendar day.
*Equity* — running total: start_balance + cumulative daily PnL.
*Peak equity* — highest equity value seen so far.
*Floor* — minimum allowed balance (trails up, locks at start_balance).
*Trailing drawdown* — peak_equity − equity (informational only).
*Max drawdown* — the distance the floor trails below the peak (e.g. $1 000).
*Profit target* — net profit needed to pass the phase (e.g. $2 000).

Layered API
-----------
1. **Low-level** — ``simulate_pnl_sequence(pnl_values, ...)``
2. **Mid-level** — ``simulate(trades, ...)``
3. **High-level** — ``analyze(trades, ...)``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class EODSnapshot:
    """Summary of the equity / drawdown state at a single day."""

    date: str               # Calendar date (ISO)
    day: int                # Trading day number (1‑based)
    daily_pnl: float        # Net PnL for this day
    cumulative_pnl: float   # Total PnL from day 1 up to this day
    equity: float           # start_balance + cumulative_pnl
    peak_equity: float      # Highest equity reached so far
    floor: float            # Minimum allowed balance for this day
    floor_locked: bool      # Whether floor is locked at start_balance
    trailing_drawdown: float  # peak_equity − equity (informational)
    dd_from_balance: float  # start_balance − equity (negative = above start)


@dataclass
class EODResult:
    """Complete result of an EOD trailing‑drawdown analysis."""

    start_balance: float
    max_drawdown_limit: float
    profit_target: float

    # ── Summary metrics ───────────────────────────────────────────────────
    passed: bool                         # Reached profit target w/o violation
    breached: bool                       # Drawdown exceeded the limit
    reason: str                          # "profit_target_reached" / "drawdown_breach" / "time_expired" / "no_data"

    final_equity: float
    peak_equity: float
    final_floor: float                   # Floor at end of simulation
    floor_locked: bool                   # Whether floor locked before end
    max_trailing_dd: float               # Worst trailing drawdown observed
    total_pnl: float

    # ── Timing ────────────────────────────────────────────────────────────
    days_traded: int                     # Number of unique calendar days processed
    day_profit_target_reached: Optional[int]  # Day number when target hit (or None)
    day_breached: Optional[int]          # Day number when drawdown breached (or None)

    # ── Raw series (useful for plotting / debugging) ─────────────────────
    snapshots: list[EODSnapshot] = field(repr=False, default_factory=list)

    def equity_series(self) -> pd.Series:
        """Return a *copy* of the daily equity curve as a pd.Series."""
        return pd.Series(
            {s.date: s.equity for s in self.snapshots},
            name="equity",
        )

    def floor_series(self) -> pd.Series:
        """Return a *copy* of the daily floor series."""
        return pd.Series(
            {s.date: s.floor for s in self.snapshots},
            name="floor",
        )

    def drawdown_series(self) -> pd.Series:
        """Return a *copy* of the daily trailing‑drawdown (informational)."""
        return pd.Series(
            {s.date: s.trailing_drawdown for s in self.snapshots},
            name="trailing_drawdown",
        )

    def daily_pnl_series(self) -> pd.Series:
        """Return a *copy* of the daily PnL series."""
        return pd.Series(
            {s.date: s.daily_pnl for s in self.snapshots},
            name="daily_pnl",
        )


# ──────────────────────────────────────────────────────────────────────────
# Low‑level: simulate from a raw sequence of daily PnL values
# ──────────────────────────────────────────────────────────────────────────


def simulate_pnl_sequence(
    pnl_values: np.ndarray | pd.Series | list[float],
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    max_daily_loss: Optional[float] = None,
    max_trading_days: Optional[int] = None,
    min_trading_days: int = 1,
) -> EODResult:
    """
    Walk through a sequence of daily PnL values and evaluate whether the
    challenge rules are met **before** the sequence is exhausted.

    Uses the correct **EOD floor-based drawdown** (Trader Launch style):
      - Floor starts at ``start_balance - max_drawdown``.
      - Floor **trails up** with peak: ``floor = max(floor, peak - max_drawdown)``.
      - Floor **locks** at ``start_balance`` once ``peak >= start_balance + max_drawdown``.
      - Breach occurs when ``equity < floor``.

    Parameters
    ----------
    pnl_values : np.ndarray | pd.Series | list[float]
        Daily net PnL values in chronological order.
    start_balance : float
        Initial account balance.
    max_drawdown : float
        Distance the floor trails below the peak (dollars).
    profit_target : float
        Net profit needed to pass the phase.
    max_daily_loss : float or None, optional
        Maximum allowed single‑day loss (dollars).  ``None`` = no limit.
    max_trading_days : int or None, optional
        Hard cap on trading days.  ``None`` = no limit.
    min_trading_days : int
        Minimum trading days before profit target is accepted (prevents
        passing in 1 day with a lucky trade).

    Returns
    -------
    EODResult
    """
    arr = np.asarray(pnl_values, dtype=float)

    if len(arr) == 0:
        return EODResult(
            start_balance=start_balance,
            max_drawdown_limit=max_drawdown,
            profit_target=profit_target,
            passed=False,
            breached=False,
            reason="no_data",
            final_equity=start_balance,
            peak_equity=start_balance,
            final_floor=start_balance - max_drawdown,
            floor_locked=False,
            max_trailing_dd=0.0,
            total_pnl=0.0,
            days_traded=0,
            day_profit_target_reached=None,
            day_breached=None,
        )

    # ---- Initial state ---------------------------------------------------
    cumulative_pnl = 0.0
    peak_equity = start_balance
    floor = start_balance - max_drawdown          # initial floor
    floor_locked = False
    max_trailing_dd = 0.0
    snapshots: list[EODSnapshot] = []

    passed = False
    breached = False
    day_target_reached: Optional[int] = None
    day_breached: Optional[int] = None

    n_days = len(arr)
    lock_threshold = start_balance + max_drawdown   # e.g. $101k

    for day_idx in range(1, n_days + 1):
        day_pnl = arr[day_idx - 1]

        # ---- Max daily loss check (triggers immediate fail) --------------
        if max_daily_loss is not None and day_pnl < -max_daily_loss:
            breached = True
            day_breached = day_idx
            snapshots.append(
                EODSnapshot(
                    date=f"day_{day_idx}",
                    day=day_idx,
                    daily_pnl=day_pnl,
                    cumulative_pnl=cumulative_pnl + day_pnl,
                    equity=start_balance + cumulative_pnl + day_pnl,
                    peak_equity=peak_equity,
                    floor=floor,
                    floor_locked=floor_locked,
                    trailing_drawdown=peak_equity - (start_balance + cumulative_pnl + day_pnl),
                    dd_from_balance=max(0.0, start_balance - (start_balance + cumulative_pnl + day_pnl)),
                )
            )
            break

        cumulative_pnl += day_pnl
        equity = start_balance + cumulative_pnl

        # ---- Update trailing peak ----------------------------------------
        if equity > peak_equity:
            peak_equity = equity

        # ---- Update floor (EOD drawdown logic) --------------------------
        if not floor_locked:
            if peak_equity >= lock_threshold:
                # Lock floor at starting balance
                floor = start_balance
                floor_locked = True
            else:
                # Floor trails up with peak (never down)
                floor = max(floor, peak_equity - max_drawdown)

        trailing_dd = peak_equity - equity
        dd_from_balance = start_balance - equity  # negative = above start

        if trailing_dd > max_trailing_dd:
            max_trailing_dd = trailing_dd

        snapshots.append(
            EODSnapshot(
                date=f"day_{day_idx}",
                day=day_idx,
                daily_pnl=day_pnl,
                cumulative_pnl=cumulative_pnl,
                equity=equity,
                peak_equity=peak_equity,
                floor=floor,
                floor_locked=floor_locked,
                trailing_drawdown=trailing_dd,
                dd_from_balance=dd_from_balance,
            )
        )

        # ---- EOD drawdown breach check: equity < floor ------------------
        if equity < floor and not passed:
            breached = True
            day_breached = day_idx
            break

        # ---- Profit target check (only after min_trading_days) ----------
        if cumulative_pnl >= profit_target and not breached:
            if day_idx >= min_trading_days:
                passed = True
                day_target_reached = day_idx
                break
            # else: keep going — need more days

        # ---- Max trading days cap ----------------------------------------
        if max_trading_days is not None and day_idx >= max_trading_days:
            break

    # ---- Determine reason ------------------------------------------------
    if passed:
        reason = "profit_target_reached"
    elif breached:
        reason = "drawdown_breach"
    else:
        reason = "time_expired"

    return EODResult(
        start_balance=start_balance,
        max_drawdown_limit=max_drawdown,
        profit_target=profit_target,
        passed=passed,
        breached=breached,
        reason=reason,
        final_equity=start_balance + cumulative_pnl,
        peak_equity=peak_equity,
        final_floor=floor,
        floor_locked=floor_locked,
        max_trailing_dd=max_trailing_dd,
        total_pnl=cumulative_pnl,
        days_traded=len(snapshots),
        day_profit_target_reached=day_target_reached,
        day_breached=day_breached,
        snapshots=snapshots,
    )


# ──────────────────────────────────────────────────────────────────────────
# Mid‑level helpers (DataFrame → daily PnL)
# ──────────────────────────────────────────────────────────────────────────


def aggregate_daily_pnl(trades: pd.DataFrame) -> pd.Series:
    """
    Aggregate trade PnL by calendar day.

    Parameters
    ----------
    trades : pd.DataFrame
        Must have a DatetimeIndex (entry timestamps) and a ``pnl`` column.

    Returns
    -------
    pd.Series
        Index is date‑only (``datetime.date``), values are daily PnL sums.
    """
    pnl = trades["pnl"]
    # Use to_series().dt.normalize() which works on both DatetimeIndex and
    # object-dtype Index of Timestamp objects.
    dates = pnl.index.to_series().dt.normalize()
    daily = pnl.groupby(dates).sum()
    daily.index = daily.index.date  # convert to datetime.date
    return daily.sort_index()


def compute_equity_curve(daily_pnl: pd.Series, start_balance: float) -> pd.Series:
    """
    Build the day‑by‑day equity curve.

    Parameters
    ----------
    daily_pnl : pd.Series
        Index by date, values are net PnL for each day.
    start_balance : float
        Starting account balance.

    Returns
    -------
    pd.Series
        Equity at the end of each day.
    """
    return start_balance + daily_pnl.cumsum()


def compute_trailing_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute trailing drawdown (peak − current) in dollar terms.

    Parameters
    ----------
    equity : pd.Series
        Equity curve indexed by date.

    Returns
    -------
    pd.Series
        Trailing drawdown at each day.
    """
    return equity.cummax() - equity


# ──────────────────────────────────────────────────────────────────────────
# Mid‑level: simulate from a trade DataFrame
# ──────────────────────────────────────────────────────────────────────────


def simulate(
    trades: pd.DataFrame,
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    max_daily_loss: Optional[float] = None,
    max_trading_days: Optional[int] = None,
    min_trading_days: int = 1,
) -> EODResult:
    """
    Aggregate trade PnL by day, then evaluate the challenge rules in
    chronological order.

    This is a **deterministic** run — data are **not** shuffled.  Use the
    Monte‑Carlo component for probabilistic analysis.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex and ``pnl`` column.
    start_balance : float
        Initial account balance.
    max_drawdown : float
        Maximum allowed trailing drawdown (dollars).
    profit_target : float
        Net profit needed to pass the phase.
    max_daily_loss : float or None, optional
        Maximum allowed single‑day loss.
    max_trading_days : int or None, optional
        Hard cap on trading days.
    min_trading_days : int
        Minimum trading days before profit target is accepted.

    Returns
    -------
    EODResult
    """
    daily_pnl = aggregate_daily_pnl(trades)
    return simulate_pnl_sequence(
        pnl_values=daily_pnl.values,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
        max_daily_loss=max_daily_loss,
        max_trading_days=max_trading_days,
        min_trading_days=min_trading_days,
    )


# ──────────────────────────────────────────────────────────────────────────
# High‑level convenience
# ──────────────────────────────────────────────────────────────────────────


def analyze(
    trades: pd.DataFrame,
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    max_daily_loss: Optional[float] = None,
    max_trading_days: Optional[int] = None,
    min_trading_days: int = 1,
) -> dict:
    """
    Convenience wrapper around :func:`simulate` that returns a plain dict
    suitable for serialisation / logging.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex and ``pnl`` column.
    start_balance : float
        Initial account balance.
    max_drawdown : float
        Maximum allowed trailing drawdown (dollars).
    profit_target : float
        Net profit needed to pass the phase.
    max_daily_loss : float or None, optional
        Maximum allowed single‑day loss.
    max_trading_days : int or None, optional
        Hard cap on trading days.
    min_trading_days : int
        Minimum trading days before profit target is accepted.

    Returns
    -------
    dict
    """
    result = simulate(
        trades=trades,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
        max_daily_loss=max_daily_loss,
        max_trading_days=max_trading_days,
        min_trading_days=min_trading_days,
    )

    return {
        "passed": result.passed,
        "breached": result.breached,
        "reason": result.reason,
        "start_balance": result.start_balance,
        "final_equity": result.final_equity,
        "peak_equity": result.peak_equity,
        "final_floor": result.final_floor,
        "floor_locked": result.floor_locked,
        "max_trailing_drawdown": result.max_trailing_dd,
        "total_pnl": result.total_pnl,
        "days_traded": result.days_traded,
        "day_profit_target_reached": result.day_profit_target_reached,
        "day_breached": result.day_breached,
        "max_drawdown_limit": result.max_drawdown_limit,
        "profit_target": result.profit_target,
    }
