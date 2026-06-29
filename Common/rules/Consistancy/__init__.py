"""
Consistency Rule Checker
========================

Reusable consistency-rule logic shared across all prop‑firm calculators.

What the 40 % consistency rule means
-------------------------------------
The rule (e.g. 40 %) states that **no single trading day's profit may exceed**
**that percentage of the total gross profits** (sum of all winning days).

This rule exists only during the initial evaluation (challenge) phase and is
dropped once the account reaches the funded stage. It is designed to filter
out gamblers — the firm wants traders who show consistent profitability over
multiple trades, not traders who get lucky on one large single trade.

Formula::

    max_daily_profit / total_gross_profits × 100 ≤ consistency_pct

Example
-------
On a $100 k account with a $2 000 profit target and a 40 % consistency rule:
no single day should make more than 40 % of your total gross profits.
If total gross profits = $5 000, then max daily profit ≤ $2 000.

When the check fails, the specific trades on the violating day are reported
so the user can see exactly which trade(s) caused the breach.

Usage
-----
>>> from Common.rules.Consistancy import check_consistency_rule
>>> result = check_consistency_rule(trades, consistency_pct=40)
>>> result["passed"]
False
>>> result["violating_trades"]   # list of offending trades
[...]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def check_consistency_rule(
    trades: pd.DataFrame,
    consistency_pct: float,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Verify the consistency rule on raw trade data.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a **DatetimeIndex** (entry timestamps) and a ``pnl``
        column.
    consistency_pct : float
        Maximum allowed percentage (e.g. ``40`` means 40 %).
    logger : logging.Logger or None
        Optional logger instance.  If ``None``, uses the module-level logger.

    Returns
    -------
    dict with keys:

    =====================  =================================================
    ``passed``             ``True`` if the rule is satisfied.
    ``max_day_pct``        Actual max day profit as a percentage of gross.
    ``max_day_date``       ISO date string of the largest profitable day (or
                           ``None``).
    ``max_day_pnl``        PnL of the largest profitable day.
    ``total_gross_profit`` Sum of all positive daily PnLs.
    ``limit_pct``          The limit that was checked against.
    ``n_violating_trades`` Number of individual trades on the violating day
                           (0 when the check passes).
    ``violating_trades``   List of trade dicts for the violating day (empty
                           when the check passes).  Each dict has keys:
                           ``entry_time``, ``exit_time``, ``side``,
                           ``quantity``, ``pnl``.
    =====================  =================================================
    """
    if logger is None:
        logger = log

    # -------------------------------------------------------------------
    # 1) Aggregate trade-level PnL to daily PnL
    # -------------------------------------------------------------------
    pnl = trades["pnl"]
    dates = pnl.index.to_series().dt.normalize()
    daily = pnl.groupby(dates).sum()
    daily.index = daily.index.date  # convert to datetime.date
    daily = daily.sort_index()

    # -------------------------------------------------------------------
    # 2) Filter to profitable days only
    # -------------------------------------------------------------------
    profits_only = daily[daily > 0]

    if len(profits_only) == 0 or profits_only.sum() <= 0:
        # No profitable days — nothing to violate
        result: dict = {
            "passed": True,
            "max_day_pct": 0.0,
            "max_day_date": None,
            "max_day_pnl": 0.0,
            "total_gross_profit": 0.0,
            "limit_pct": consistency_pct,
            "n_violating_trades": 0,
            "violating_trades": [],
        }
        logger.info("  Consistency rule: PASS  (no profitable days)")
        return result

    total_gross = profits_only.sum()
    max_pnl = profits_only.max()
    max_date = profits_only.idxmax()          # datetime.date object
    max_pct = 100.0 * max_pnl / total_gross
    passed = max_pct <= consistency_pct

    # -------------------------------------------------------------------
    # 3) Collect the individual trades on the violating day
    # -------------------------------------------------------------------
    # Filter trades whose normalized date matches max_date
    violating_trades = trades[
        trades.index.to_series().dt.normalize().dt.date == max_date
    ]

    trade_details: list[dict] = []
    for idx, row in violating_trades.iterrows():
        trade_details.append({
            "entry_time": str(idx),
            "exit_time": str(row.get("exit_time", "")),
            "side": str(row.get("side", "")),
            "quantity": int(row.get("quantity", 0)),
            "pnl": float(row.get("pnl", 0)),
        })

    result = {
        "passed": passed,
        "max_day_pct": round(max_pct, 2),
        "max_day_date": str(max_date),
        "max_day_pnl": round(max_pnl, 2),
        "total_gross_profit": round(total_gross, 2),
        "limit_pct": consistency_pct,
        "n_violating_trades": len(trade_details),
        "violating_trades": trade_details,
    }

    # -------------------------------------------------------------------
    # 4) Log result — on failure, print each offending trade
    # -------------------------------------------------------------------
    if passed:
        logger.info(
            "  Consistency rule: PASS  (max day %.1f%% ≤ %.0f%% limit)",
            max_pct, consistency_pct,
        )
    else:
        logger.warning(
            "  Consistency rule: FAIL  (max day %.1f%% > %.0f%% limit)",
            max_pct, consistency_pct,
        )
        logger.warning(
            "    →  Violating day: %s  |  Day PnL: $%.2f  |  "
            "%.1f%% of $%.2f total gross profits",
            max_date, max_pnl, max_pct, total_gross,
        )
        logger.warning("    →  Individual trades on that day (%d trades):", len(trade_details))
        for td in trade_details:
            logger.warning(
                "       [%s]  %-5s  qty=%-2d  pnl=$%+.2f",
                td["entry_time"], td["side"], td["quantity"], td["pnl"],
            )

    return result
