"""
Trader Launch Challenge Phase — Pre‑check Rules Filter

Examines raw CSV trade data for apparent / data‑level rule violations
before the full Challenge Phase calculation runs.

Rules checked (from rule_sets_metadata.yaml):
  1. Min trading days         — at least N unique calendar days
  2. Max contracts per trade  — no single trade exceeds the limit
  3. Inactivity rule          — no gap longer than N days between trading days

If any check fails the pipeline is halted.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

log = logging.getLogger("TraderLaunch.RulesFilter.Challenge")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RULES_DIR = Path(__file__).resolve().parents[1] / "rules"
RULES_YAML = RULES_DIR / "rule_sets_metadata.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_rules(yaml_path: Union[str, Path] = RULES_YAML) -> dict:
    """Load rule-set metadata from the YAML file."""
    with open(yaml_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_min_trading_days(trades: pd.DataFrame, min_days: int) -> dict:
    """
    Pre‑check: at least ``min_days`` unique calendar days with trades.
    """
    unique_days = pd.Series(trades.index).dt.normalize().unique()
    actual = len(unique_days)
    passed = actual >= min_days
    msg = "✔  PASS" if passed else "✘  FAIL"
    log.info("  Min trading days     —  need ≥ %d, got %d  [%s]", min_days, actual, msg)
    return {
        "rule": "min_trading_days",
        "required": min_days,
        "actual": actual,
        "passed": passed,
        "days_with_trades": sorted(d.date() for d in unique_days),
    }


def check_max_contracts(trades: pd.DataFrame, max_contracts: int) -> dict:
    """
    Pre‑check: no single trade exceeds ``max_contracts`` contracts.

    Uses the ``quantity`` column from the CSV.
    """
    over_limit = trades[trades["quantity"] > max_contracts]
    passed = len(over_limit) == 0
    if passed:
        log.info("  Max contracts/trade  —  ✔  PASS  (limit = %d)", max_contracts)
    else:
        log.info("  Max contracts/trade  —  ✘  FAIL  (%d trade(s) over limit)", len(over_limit))
        for idx, row in over_limit.iterrows():
            log.info("      %s  quantity=%d", idx, row["quantity"])
    return {
        "rule": "max_contracts_per_trade",
        "required": max_contracts,
        "violations": len(over_limit),
        "violation_details": [
            {"time": str(idx), "quantity": int(row["quantity"])}
            for idx, row in over_limit.iterrows()
        ],
        "passed": passed,
    }


def check_inactivity(trades: pd.DataFrame, max_gap_days: int) -> dict:
    """
    Pre‑check: no gap longer than ``max_gap_days`` between consecutive
    trading days.
    """
    unique_dates = pd.Series(trades.index).dt.normalize().unique()
    sorted_dates = sorted(unique_dates)
    gaps: list[dict] = []
    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        if gap > max_gap_days:
            gaps.append({
                "from": str(sorted_dates[i - 1].date()),
                "to": str(sorted_dates[i].date()),
                "gap_days": gap,
            })
    passed = len(gaps) == 0
    if passed:
        log.info("  Inactivity           —  ✔  PASS  (no gaps > %d days)", max_gap_days)
    else:
        log.info("  Inactivity           —  ✘  FAIL  (%d gap(s) > %d days)", len(gaps), max_gap_days)
        for g in gaps:
            log.info("      %s → %s : %d days", g["from"], g["to"], g["gap_days"])
    return {
        "rule": "inactivity_rule_day",
        "max_gap_days": max_gap_days,
        "gaps": gaps,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run(
    trades: pd.DataFrame,
    config: Optional[dict] = None,
) -> dict:
    """
    Run the three pre‑checks against raw trade data.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex (entry_time) and a 'pnl' column.
    config : dict or None
        Optional rule overrides.  If None, rules are loaded from the YAML file.

    Returns
    -------
    dict with keys:
        'all_passed'  — bool
        'checks'      — list of per-rule result dicts
        'rule_set'    — the rule configuration that was used
    """
    log.info("─" * 55)
    log.info("  Challenge Phase — Pre‑check")
    log.info("─" * 55)

    # ---- Load rules -------------------------------------------------------
    rules = _load_rules() if config is None else config
    challenge_cfg = rules["phases"]["challenge"]

    checks: list[dict] = []

    # 1) Min trading days ---------------------------------------------------
    checks.append(check_min_trading_days(trades, challenge_cfg["min_trading_days"]))

    # 2) Max contracts per trade -------------------------------------------
    checks.append(
        check_max_contracts(trades, challenge_cfg["maximum_contracts_per_trade"])
    )

    # 3) Inactivity gap -----------------------------------------------------
    checks.append(
        check_inactivity(trades, rules.get("general", {}).get("inactivity_rule_day", 5))
    )

    # ---- Summary ----------------------------------------------------------
    all_passed = all(c["passed"] for c in checks)

    log.info("─" * 55)
    if all_passed:
        log.info("  ✅  ALL PRE‑CHECKS PASSED — proceeding to Challenge Phase")
    else:
        log.info("  ❌  SOME PRE‑CHECKS FAILED — halting pipeline")
    log.info("─" * 55)

    result = {
        "all_passed": all_passed,
        "checks": checks,
        "rule_set": challenge_cfg,
    }

    if not all_passed:
        raise RuntimeError(
            "Challenge Phase — Pre‑check FAILED. See logs above for details."
        )

    return result
