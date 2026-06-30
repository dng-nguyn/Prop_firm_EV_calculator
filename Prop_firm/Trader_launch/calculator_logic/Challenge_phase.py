"""
Trader Launch — Challenge Phase Calculation
============================================

Evaluates trade data against Trader Launch challenge-phase rules.

Pipeline step context
---------------------
Called **after** the rules filter (step 1) has verified that the raw trade
data passes structural checks (min days, max contracts, inactivity).

Uses
----
- ``Common.EOD``           — low-level equity / trailing-drawdown engine
- ``Common.Monte_carlo``   — shuffled‑sequence pass‑rate estimation

Outputs
-------
- ``Challenge_phase.txt``  — human‑readable summary
- ``Challenge_phase_calculation_plot.png`` — equity curve + drawdown chart

Returns (dict)
--------------
Deterministic run + Monte Carlo results + consistency‑rule check.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``Common`` is importable
# regardless of which directory the user runs the calculator from.
# This MUST happen before any ``from Common.xxx import ...`` statements.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Prop_firm_calculator/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from Common.EOD.trailing_drawdown import (
    EODResult,
    aggregate_daily_pnl,
    simulate,
)
from Common.Monte_carlo.simulator import (
    SimulationResult,
    run_simulations,
)
from Common.rules.Consistancy import check_consistency_rule

log = logging.getLogger("TraderLaunch.Challenge")


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


def apply_daily_lock_rule(
    trades: pd.DataFrame,
    profit_target: float,
    consistency_pct: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Enforce the Trader Launch daily consistency lock on raw trade data.

    For each trading day, scan trades chronologically.  If at any point the
    day's running realized PnL **plus** the current trade's highest unrealized
    profit reaches or exceeds the lock threshold, the day is "locked":

      - The current trade **is** kept (its realized PnL counts).
      - All **subsequent** trades on that same day are discarded.

    The lock threshold is::

        lock_amount = profit_target * (consistency_pct / 100)

    For a $100k account with a $2 000 target and 40 % rule, this is $800.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex (entry timestamps), a ``pnl`` column,
        and a ``highest_unrealized_profit`` column.
    profit_target : float
        Phase profit target (e.g. 2000.0).
    consistency_pct : float
        Consistency rule percentage (e.g. 40 for 40 %).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (filtered_trades, stats) where ``stats`` contains:

        ==================  ================================================
        ``lock_amount``     Threshold used (profit_target × pct / 100).
        ``locked_days``     Number of days where the lock triggered.
        ``total_days``      Number of unique trading days processed.
        ``trades_removed``  Count of post-lock trades discarded.
        ``trades_kept``     Count of trades retained.
        ``triggered``       Whether any day was locked.
        ==================  ================================================
    """
    lock_amount = profit_target * (consistency_pct / 100.0)

    # If the column is missing, skip the filter gracefully
    if "highest_unrealized_profit" not in trades.columns:
        log.warning(
            "  Column 'highest_unrealized_profit' not found — "
            "skipping daily lock rule filter."
        )
        return trades, {
            "lock_amount": lock_amount,
            "locked_days": 0,
            "total_days": 0,
            "trades_removed": 0,
            "trades_kept": len(trades),
            "triggered": False,
        }

    # Work on a copy so we can cap PnL values without mutating the original
    result_trades = trades.copy()
    keep_mask = pd.Series(True, index=result_trades.index)

    # Group by calendar day
    day_labels = result_trades.index.to_series().dt.normalize()
    locked_days: int = 0
    total_days: int = 0
    locked_day_dates: list[str] = []

    for day, group_idx in day_labels.groupby(day_labels):
        total_days += 1
        day_trades = result_trades.loc[group_idx.index]

        # Sort within the day by entry time (already sorted, but be safe)
        day_trades = day_trades.sort_index()

        running_day_pnl = 0.0
        day_locked = False

        for entry_time, trade in day_trades.iterrows():
            if day_locked:
                keep_mask.loc[entry_time] = False
                continue

            trade_pnl = float(trade["pnl"])
            highest_unrealized = float(trade.get("highest_unrealized_profit", 0.0))

            # Check if this trade triggers the lock
            if running_day_pnl + highest_unrealized >= lock_amount:
                day_locked = True
                locked_days += 1
                locked_day_dates.append(str(day.date()))

                # Cap this trade's PnL so the day total = lock_amount
                room_to_lock = lock_amount - running_day_pnl
                capped_pnl = min(trade_pnl, room_to_lock)
                result_trades.loc[entry_time, "pnl"] = capped_pnl

                log.info(
                    "    Day lock triggered: %s  |  "
                    "running PnL $%.2f + unrealized $%.0f >= $%.0f  |  "
                    "trade PnL $%+.2f → capped to $%.2f",
                    day.date(), running_day_pnl, highest_unrealized,
                    lock_amount, trade_pnl, capped_pnl,
                )
                # Keep this trade (capped), discard subsequent ones
                continue

            running_day_pnl += trade_pnl

    n_removed = int((~keep_mask).sum())
    n_kept = len(trades) - n_removed

    if n_removed > 0:
        log.info(
            "  Daily lock rule: %d trade(s) removed across %d locked day(s) / %d day(s)",
            n_removed, locked_days, total_days,
        )
    elif total_days > 0:
        log.info(
            "  Daily lock rule: no trades removed (no day hit the $%.0f threshold)",
            lock_amount,
        )

    stats = {
        "lock_amount": lock_amount,
        "locked_days": locked_days,
        "total_days": total_days,
        "trades_removed": n_removed,
        "trades_kept": n_kept,
        "triggered": locked_days > 0,
        "locked_day_dates": locked_day_dates,
    }

    return result_trades.loc[keep_mask], stats


def _format_result_text(
    eod_result: EODResult,
    mc_result: SimulationResult,
    consistency: dict,
    lock_rule_stats: dict,
    challenge_cfg: dict,
    general_cfg: dict,
    start_balance: float,
    max_trading_days: int,
    eval_fee: float,
    daily_pnl: Optional[pd.Series] = None,
) -> str:
    """Build the human-readable Challenge_phase.txt content."""
    lines = [
        "# Trader Launch — Challenge Phase Results",
        "# ========================================",
        "",
        "─── Challenge Configuration ───",
        f"  Firm:                       Trader Launch",
        f"  Account size:               ${start_balance:,.0f}",
        f"  Evaluation fee:             ${eval_fee:,.0f}",
        f"  Activation fee:             ${general_cfg.get('activation_fee', 0):,.0f}",
        "",
        "─── Phase Criteria ───",
    ]

    # Build a structured criteria table
    lock_threshold = eod_result.start_balance + eod_result.max_drawdown_limit
    criteria = [
        ("Profit target", f"${challenge_cfg['profit_target']:,.0f}",
         f"${eod_result.total_pnl:,.2f}", eod_result.passed),
        ("EOD drawdown (equity < floor)", f"Must stay above floor",
         f"Floor=${eod_result.final_floor:,.0f}, Min equity=${min(s.equity for s in eod_result.snapshots):,.0f}",
         not eod_result.breached),
        ("Min trading days", str(challenge_cfg['min_trading_days']),
         str(eod_result.days_traded), eod_result.days_traded >= challenge_cfg['min_trading_days']),
        ("Max trading days (initial)", str(max_trading_days),
         str(eod_result.days_traded) + " used", eod_result.days_traded <= max_trading_days),
        ("Consistency rule", f"{challenge_cfg['consistency_rule']}% max single day",
         f"{consistency['max_day_pct']}%", consistency['passed']),
        ("Max contracts per trade", str(challenge_cfg['maximum_contracts_per_trade']),
         "checked in rules filter", True),
    ]

    for name, required, actual, passed in criteria:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {name}")
        lines.append(f"         Required: {required}")
        lines.append(f"         Actual:   {actual}")

    lines += [
        "",
        "─── EOD Drawdown Floor Details ───",
        f"  Initial floor:              ${eod_result.start_balance - eod_result.max_drawdown_limit:,.0f}  (start - max_drawdown)",
        f"  Final floor:                ${eod_result.final_floor:,.0f}",
        f"  Floor locked:               {eod_result.floor_locked}",
        f"  Lock threshold:             ${lock_threshold:,.0f}  (start + max_drawdown)",
    ]
    if eod_result.floor_locked:
        lock_day = next((s.day for s in eod_result.snapshots if s.floor_locked), None)
        lock_date = ""
        if lock_day and daily_pnl is not None and lock_day <= len(daily_pnl):
            lock_date = f"  ({daily_pnl.index[lock_day - 1]})"
        lines.append(f"  Floor locked on day:        {lock_day}{lock_date}")

    lines += [
        "",
        "─── Detailed Metrics ───",
        f"  Start balance:              ${eod_result.start_balance:,.2f}",
        f"  Final equity:               ${eod_result.final_equity:,.2f}",
        f"  Peak equity:                ${eod_result.peak_equity:,.2f}",
        f"  Total PnL:                  ${eod_result.total_pnl:,.2f}",
        f"  Max trailing drawdown:      ${eod_result.max_trailing_dd:,.2f}  (peak − equity, informational)",
        f"  Max drawdown parameter:     ${eod_result.max_drawdown_limit:,.0f}  (floor trails this far below peak)",
        f"  Profit target:              ${eod_result.profit_target:,.0f}",
        "",
        "─── Deterministic Run ───",
        f"  Result:                     {eod_result.reason}",
        f"  Passed:                     {eod_result.passed}",
        f"  Breached:                   {eod_result.breached}",
        f"  Days traded:                {eod_result.days_traded}",
        f"  Day profit target reached:  {eod_result.day_profit_target_reached}"
        + (f"  ({daily_pnl.index[eod_result.day_profit_target_reached - 1]})"
           if eod_result.day_profit_target_reached and daily_pnl is not None
           and eod_result.day_profit_target_reached <= len(daily_pnl) else ""),
        f"  Day drawdown breached:      {eod_result.day_breached}"
        + (f"  ({daily_pnl.index[eod_result.day_breached - 1]})"
           if eod_result.day_breached and daily_pnl is not None
           and eod_result.day_breached <= len(daily_pnl) else ""),
        "",
        "─── Consistency Rule Detail ───",
        f"  Passed:                     {consistency['passed']}",
        f"  Max day profit (%):         {consistency['max_day_pct']}%",
        f"  Max day date:               {consistency['max_day_date']}",
    ]
    if consistency["max_day_pnl"]:
        lines.append(f"  Max day PnL:                ${consistency['max_day_pnl']:,.2f}")
    else:
        lines.append("  Max day PnL:                $0.00")
    if consistency["total_gross_profit"]:
        lines.append(f"  Total gross profit:         ${consistency['total_gross_profit']:,.2f}")
    else:
        lines.append("  Total gross profit:         $0.00")
    lines.append(f"  Limit:                      {consistency.get('limit_pct', 40)}%")

    # Show violating trades when the rule fails
    if not consistency["passed"] and consistency.get("violating_trades"):
        lines.append("")
        lines.append("  ⚠  Violating trades on the offending day:")
        for vt in consistency["violating_trades"]:
            lines.append(
                f"      [{vt['entry_time']}]  {vt['side']:<5}  "
                f"qty={vt['quantity']:<2}  pnl=${vt['pnl']:+.2f}"
            )

    # ── Daily Lock Rule section ──────────────────────────────────────────
    lines += [
        "",
        "─── Daily Lock Rule Detail ───",
        f"  Triggered:                  {lock_rule_stats['triggered']}",
        f"  Lock threshold:             ${lock_rule_stats['lock_amount']:,.0f}  "
        f"(profit_target × {challenge_cfg.get('consistency_rule', 40)}%)",
        f"  Locked days:                {lock_rule_stats['locked_days']} / {lock_rule_stats['total_days']}",
        f"  Trades removed:             {lock_rule_stats['trades_removed']}",
        f"  Trades kept:                {lock_rule_stats['trades_kept']}",
    ]

    lines += [
        "",
        "─── Monte Carlo Simulation ───",
        f"  Number of simulations:      {mc_result.n_simulations:,}",
        f"  Pass rate:                  {mc_result.pass_rate:.2f}%",
        f"  Fail rate (drawdown):       {mc_result.fail_rate:.2f}%",
        f"  Time expired rate:          {mc_result.time_expired_rate:.2f}%",
        "",
    ]
    if mc_result.avg_pass_days is not None:
        lines.append(f"  Avg pass days:              {mc_result.avg_pass_days:.1f}")
    else:
        lines.append("  Avg pass days:              N/A")
    if mc_result.median_pass_days is not None:
        lines.append(f"  Median pass days:           {mc_result.median_pass_days}")
    else:
        lines.append("  Median pass days:           N/A")
    if mc_result.avg_fail_days is not None:
        lines.append(f"  Avg fail days:              {mc_result.avg_fail_days:.1f}")
    else:
        lines.append("  Avg fail days:              N/A")

    lines += [
        f"  Avg max drawdown:           ${mc_result.avg_max_drawdown:,.2f}",
        f"  Avg final PnL:              ${mc_result.avg_final_pnl:,.2f}",
        f"  Evaluation fee:             ${eval_fee:,.2f}",
        "",
        "─── Verdict ───",
    ]

    # Determine overall verdict
    all_criteria_pass = all(c[3] for c in criteria)
    if eod_result.passed and consistency["passed"] and all_criteria_pass:
        lines.append("  OUTCOME: CHALLENGE PASSED")
        lines.append("  All criteria met on the deterministic run.")
    elif eod_result.passed:
        lines.append("  OUTCOME: CHALLENGE PASSED (deterministic) - review consistency/rules")
    else:
        lines.append("  OUTCOME: CHALLENGE FAILED - review criteria above")

    # Monte Carlo verdict
    if mc_result.pass_rate >= 50.0:
        lines.append("  Sequence risk: LOW (MC pass rate >= 50%)")
    elif mc_result.pass_rate >= 25.0:
        lines.append("  Sequence risk: MODERATE (MC pass rate 25-50%)")
    else:
        lines.append("  Sequence risk: HIGH (MC pass rate < 25%)")

    lines += [
        "",
        "─── Notes ───",
        "  - The deterministic run evaluates trades in their actual chronological order.",
        "  - The Monte Carlo simulation shuffles daily PnLs to assess sequence risk.",
        "  - Low MC pass rate + deterministic pass = strategy is order-dependent.",
        "  - High MC pass rate = strategy is robust regardless of PnL ordering.",
        "",
        "─── EOD Drawdown Behaviour ───",
        "  - Floor starts at start_balance - max_drawdown (e.g. $99k).",
        "  - Floor trails UP when you make profits: floor = peak - max_drawdown.",
        "  - Floor NEVER moves down when you lose money.",
        "  - Floor LOCKS at start_balance ($100k) once peak >= start + max_drawdown ($101k).",
        "  - After locking, available drawdown room can exceed max_drawdown.",
        "  - Breach occurs when equity < floor (NOT when trailing_dd > max_drawdown).",
    ]
    return "\n".join(lines)


def _generate_plot(
    eod_result: EODResult,
    daily_pnl: pd.Series,
    save_path: Path,
    lock_rule_stats: Optional[dict] = None,
) -> None:
    """
    Generate and save a two-panel plot:
      1. Equity curve with floor (EOD drawdown limit), peak, target lines.
      2. Daily PnL bar chart (with optional daily lock rule markers).

    Parameters
    ----------
    eod_result : EODResult
        Result from the deterministic EOD run.
    daily_pnl : pd.Series
        Daily PnL values (index by date). Used to match locked days.
    save_path : Path
        Where to save the PNG.
    lock_rule_stats : dict or None
        Optional stats from ``apply_daily_lock_rule``. If provided, the plot
        will mark days where the daily lock was triggered and show the
        lock threshold line.
    """
    dates = [s.date for s in eod_result.snapshots]
    equities = [s.equity for s in eod_result.snapshots]
    peak = [s.peak_equity for s in eod_result.snapshots]
    floors = [s.floor for s in eod_result.snapshots]
    drawdowns = [s.trailing_drawdown for s in eod_result.snapshots]
    daily_pnls = [s.daily_pnl for s in eod_result.snapshots]
    floor_locked = eod_result.floor_locked

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Trader Launch — Challenge Phase", fontsize=14, fontweight="bold")

    # Prepend day 0 (starting point before any trades)
    x = list(range(1, len(dates) + 1))
    start_bal = eod_result.start_balance
    x_ext = [0] + x
    equities_ext = [start_bal] + equities
    peak_ext = [start_bal] + peak
    # Floor at day 0: start_balance - max_drawdown (e.g. $99k)
    floor_day0 = start_bal - eod_result.max_drawdown_limit
    floors_ext = [floor_day0] + floors

    x_labels = dates
    step = max(1, len(x_ext) // 15)

    lock_threshold = start_bal + eod_result.max_drawdown_limit  # $101k

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — Equity curve with floor
    # ═══════════════════════════════════════════════════════════════════════
    ax1.plot(x_ext, equities_ext, "b-", linewidth=1.8, label="Equity")
    ax1.plot(x_ext, peak_ext, "g--", linewidth=1.0, alpha=0.7, label="Peak equity")

    # Floor line (stepped — the actual EOD drawdown limit, starts at start - max_drawdown)
    ax1.step(x_ext, floors_ext, "r-", linewidth=2.0, where="pre", label="Floor (EOD drawdown limit)")
    # Shade the danger zone (below floor)
    ax1.fill_between(x_ext, floors_ext, min(floors_ext) - 500, alpha=0.08, color="red")

    ax1.axhline(eod_result.start_balance, color="gray", linestyle=":", alpha=0.6, label="Start balance ($100k)")
    ax1.axhline(
        eod_result.start_balance + eod_result.profit_target,
        color="green", linestyle="--", alpha=0.5, label=f"Target (${eod_result.profit_target:,.0f})",
    )
    ax1.axhline(lock_threshold, color="orange", linestyle="-.", alpha=0.4,
                label=f"Lock threshold (${lock_threshold:,.0f})")

    # Annotate floor lock
    if floor_locked:
        lock_day = next((i + 1 for i, s in enumerate(eod_result.snapshots) if s.floor_locked), None)
        if lock_day:
            ax1.annotate(
                f"Floor locked\nat $100k",
                xy=(lock_day, eod_result.start_balance),
                xytext=(lock_day + 0.5, eod_result.start_balance + 300),
                fontsize=9, color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred"),
            )

    if eod_result.day_profit_target_reached:
        ax1.axvline(eod_result.day_profit_target_reached, color="green", linestyle=":", alpha=0.6)
        ax1.annotate(
            f"Target hit\nday {eod_result.day_profit_target_reached}",
            xy=(eod_result.day_profit_target_reached, eod_result.final_equity),
            xytext=(10, -30), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=9, color="green",
        )

    ax1.set_ylabel("Account Equity ($)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Equity Curve with EOD Floor (drawdown limit trails up, locks at $100k)")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Daily PnL
    # ═══════════════════════════════════════════════════════════════════════
    colors = ["green" if v >= 0 else "red" for v in daily_pnls]
    ax3.bar(x, daily_pnls, color=colors, width=0.7, alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_xlabel("Trading Day")
    ax3.set_ylabel("Daily PnL ($)")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Daily PnL Breakdown")

    all_labels = ["Start"] + list(x_labels)
    tick_positions = x_ext[::step]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([all_labels[i] for i in range(0, len(x_ext), step)], rotation=45, ha="right", fontsize=8)

    # ═══════════════════════════════════════════════════════════════════════
    # Daily lock rule markers  (optional — on Panel 2)
    # ═══════════════════════════════════════════════════════════════════════
    if lock_rule_stats is not None and lock_rule_stats.get("triggered"):
        lock_amount = lock_rule_stats["lock_amount"]
        locked_day_dates = lock_rule_stats.get("locked_day_dates", [])

        # Horizontal dashed line at the lock threshold
        ax3.axhline(
            lock_amount, color="orange", linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Daily lock threshold (${lock_amount:,.0f})",
        )

        # Map locked calendar dates to snapshot x-positions.
        # The daily_pnl Series has the actual calendar dates as index;
        # locked_day_dates are ISO date strings (e.g. "2026-01-20").
        daily_pnl_dates = [str(d) for d in daily_pnl.index]
        locked_x_positions = []
        for locked_date in locked_day_dates:
            if locked_date in daily_pnl_dates:
                # snapshots may be shorter than daily_pnl (if simulation ended early)
                idx = daily_pnl_dates.index(locked_date)
                if idx < len(x):
                    locked_x_positions.append(x[idx])

        if locked_x_positions:
            # Vertical markers on locked days
            for lx in locked_x_positions:
                ax3.axvline(lx, color="orange", linestyle=":", linewidth=1.2, alpha=0.7)

            # Shade locked day regions
            for lx in locked_x_positions:
                ax3.axvspan(
                    lx - 0.4, lx + 0.4,
                    color="orange", alpha=0.08,
                )

            # Annotation near the last visible locked day
            last_lx = locked_x_positions[-1]
            ax3.annotate(
                f"[LOCK] Day locked\nthreshold hit",
                xy=(last_lx, lock_amount),
                xytext=(last_lx + 1.5, lock_amount * 1.7),
                fontsize=8, color="darkorange", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkorange", alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        ax3.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    trades: pd.DataFrame,
    config: Optional[dict] = None,
    reports_dir: Optional[Union[str, Path]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    n_mc_simulations: int = 5_000,
    mc_seed: Optional[int] = None,
    mc_block_size: int = 0,  # 0 = auto-select
    mc_circular: bool = False,
    mc_floor_aware: bool = False,
) -> dict:
    """
    Run the Trader Launch Challenge Phase analysis.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with a DatetimeIndex and ``pnl`` column.
    config : dict or None
        Optional rule overrides.  If None, rules are loaded from the YAML
        metadata file.
    reports_dir : str or Path or None
        Directory for text reports.  If None, no report is written.
    figures_dir : str or Path or None
        Directory for plots.  If None, no plot is generated.
    n_mc_simulations : int
        Number of Monte Carlo shuffled simulations.
    mc_seed : int or None
        Random seed for Monte Carlo reproducibility.

    Returns
    -------
    dict with keys:
        deterministic_run  — EODResult (dataclass)
        monte_carlo        — SimulationResult (dataclass)
        consistency        — dict from consistency rule check
        daily_pnl          — pd.Series of daily PnL
        summary            — plain dict for downstream aggregation
    """
    log.info("─" * 55)
    log.info("  Trader Launch — Challenge Phase Calculation")
    log.info("─" * 55)

    # ---- Load rules -------------------------------------------------------
    rules = _load_rules() if config is None else config
    challenge_cfg = rules["phases"]["challenge"]
    general_cfg = rules.get("general", {})

    start_balance = 100_000.0        # default 100k account
    max_drawdown = float(challenge_cfg["max_total_drawdown"])
    profit_target = float(challenge_cfg["profit_target"])
    min_trading_days = int(challenge_cfg["min_trading_days"])
    max_trading_days = int(challenge_cfg["max_trading_days_initial"])
    consistency_pct = float(challenge_cfg["consistency_rule"])
    eval_fee = float(general_cfg.get("evaluation_fee", 45))

    # -----------------------------------------------------------------------
    # 0) Apply daily lock rule  (filter out post-lock intraday trades)
    # -----------------------------------------------------------------------
    trades_filtered, lock_rule_stats = apply_daily_lock_rule(
        trades=trades,
        profit_target=profit_target,
        consistency_pct=consistency_pct,
    )

    # -----------------------------------------------------------------------
    # 1) Build daily PnL  (from filtered trades)
    # -----------------------------------------------------------------------
    daily_pnl = aggregate_daily_pnl(trades_filtered)
    log.info("  Unique trading days: %d", len(daily_pnl))
    log.info("  Total PnL: ${:,.2f}".format(daily_pnl.sum()))

    # -----------------------------------------------------------------------
    # 2) Deterministic EOD run (actual chronological order)
    # -----------------------------------------------------------------------
    eod_result = simulate(
        trades=trades_filtered,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
        max_trading_days=max_trading_days,
        min_trading_days=min_trading_days,
    )

    log.info("  Deterministic run: %s", eod_result.reason)
    if eod_result.passed:
        log.info("    -> PASSED  (day %d, max DD $%.2f)", eod_result.day_profit_target_reached, eod_result.max_trailing_dd)
    elif eod_result.breached:
        log.info("    -> BREACH  (day %d, max DD $%.2f)", eod_result.day_breached, eod_result.max_trailing_dd)
    else:
        log.info("    -> TIME EXPIRED  (max DD $%.2f)", eod_result.max_trailing_dd)

    # -----------------------------------------------------------------------
    # 3) Consistency rule check  (delegated to Common.rules.Consistancy)
    # -----------------------------------------------------------------------
    consistency = check_consistency_rule(trades_filtered, consistency_pct, logger=log)

    # -----------------------------------------------------------------------
    # 4) Monte Carlo simulation
    # -----------------------------------------------------------------------
    log.info("  Running Monte Carlo (%d simulations)...", n_mc_simulations)
    mc_result = run_simulations(
        daily_pnl=daily_pnl,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
        n_simulations=n_mc_simulations,
        max_trading_days=max_trading_days,
        min_trading_days=min_trading_days,
        seed=mc_seed,
        block_size=mc_block_size,
        circular=mc_circular,
        floor_aware=mc_floor_aware,
    )
    log.info("  MC pass rate: %.1f%%  (fail: %.1f%%)", mc_result.pass_rate, mc_result.fail_rate)

    # -----------------------------------------------------------------------
    # 5) Write text report
    # -----------------------------------------------------------------------
    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        report_path = reports_path / "Challenge_phase.txt"

        report_text = _format_result_text(
            eod_result=eod_result,
            mc_result=mc_result,
            consistency=consistency,
            lock_rule_stats=lock_rule_stats,
            challenge_cfg=challenge_cfg,
            general_cfg=general_cfg,
            start_balance=start_balance,
            max_trading_days=max_trading_days,
            eval_fee=eval_fee,
            daily_pnl=daily_pnl,
        )
        report_path.write_text(report_text, encoding="utf-8")
        log.info("  Report written to %s", report_path)

    # -----------------------------------------------------------------------
    # 6) Generate plot
    # -----------------------------------------------------------------------
    if figures_dir is not None:
        figures_path = Path(figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        plot_path = figures_path / "Challenge_phase_calculation_plot.png"
        _generate_plot(
            eod_result, daily_pnl, plot_path,
            lock_rule_stats=lock_rule_stats,
        )

    # -----------------------------------------------------------------------
    # Return aggregated results
    # -----------------------------------------------------------------------
    return {
        "deterministic_run": eod_result,
        "monte_carlo": mc_result,
        "consistency": consistency,
        "daily_lock_rule": lock_rule_stats,
        "daily_pnl": daily_pnl,
        "summary": {
            "deterministic_passed": eod_result.passed,
            "deterministic_reason": eod_result.reason,
            "deterministic_days": eod_result.days_traded,
            "deterministic_day_target": eod_result.day_profit_target_reached,
            "deterministic_max_dd": round(eod_result.max_trailing_dd, 2),
            "consistency_passed": consistency["passed"],
            "consistency_max_day_pct": consistency["max_day_pct"],
            "mc_pass_rate_pct": round(mc_result.pass_rate, 2),
            "mc_fail_rate_pct": round(mc_result.fail_rate, 2),
            "mc_avg_pass_days": round(mc_result.avg_pass_days, 1) if mc_result.avg_pass_days is not None else None,
            "mc_n_simulations": mc_result.n_simulations,
        },
    }
