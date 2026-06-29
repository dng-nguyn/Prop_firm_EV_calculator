"""
Trader Launch Funded Phase — Rules Filter
==========================================

The funded phase has **no additional rules** beyond the EOD trailing drawdown.
The consistency rule (40%) and daily lock rule are dropped once the account
reaches the funded stage.

This filter is a pass-through — all trades are accepted.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger("TraderLaunch.RulesFilter.Funded")


def run(
    trades: pd.DataFrame,
    config: Optional[dict] = None,
) -> dict:
    """
    Funded phase rules filter — pass-through (no additional rules).

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex.
    config : dict or None
        Optional configuration (not used in funded phase).

    Returns
    -------
    dict with keys:
        'all_passed'  — always True
        'checks'      — empty list
    """
    log.info("─" * 55)
    log.info("  Funded Phase — Pre‑check")
    log.info("─" * 55)
    log.info("  No additional rules in funded phase — all trades accepted.")
    log.info("─" * 55)
    log.info("  ✅  ALL PRE‑CHECKS PASSED — proceeding to Funded Phase")
    log.info("─" * 55)

    return {
        "all_passed": True,
        "checks": [],
        "note": "Funded phase has no additional rules beyond EOD drawdown.",
    }
