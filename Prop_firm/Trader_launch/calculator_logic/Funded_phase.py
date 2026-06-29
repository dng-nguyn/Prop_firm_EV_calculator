"""
Trader Launch — Funded Phase Calculation
=========================================

Evaluates trade data against funded-phase rules after the challenge is passed.

Key differences from Challenge phase
------------------------------------
- No consistency rule (40% rule is dropped)
- No daily lock rule
- No minimum trading days
- Profit target is lower ($1 000 vs $2 000)
- Account resets to $100 000 start balance
- Trades are filtered to those occurring **after** the challenge pass date

Pipeline context
----------------
Called after the Funded Phase Rules Filter (step 3) and after the Challenge
Phase calculation (step 2) has completed successfully.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
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

log = logging.getLogger("TraderLaunch.Funded")


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


def _format_result_text(
    eod_result: EODResult,
    mc_result: Optional[SimulationResult],
    funded_cfg: dict,
    start_balance: float,
    challenge_passed_date: str,
    total_pnl_in_funded: float,
    days_traded_in_funded: int,
    daily_pnl: Optional[pd.Series] = None,
) -> str:
    """Build the human-readable Funded_phase.txt content."""
    lines = [
        "# Trader Launch — Funded Phase Results",
        "# ======================================",
        "",
        "─── Funded Phase Configuration ───",
        f"  Firm:                       Trader Launch",
        f"  Challenge passed on:        {challenge_passed_date}",
        f"  Starting balance:           ${start_balance:,.2f}",
        "",
        "─── Phase Criteria ───",
    ]

    criteria = [
        ("Profit target", f"${funded_cfg['profit_target']:,.0f}",
         f"${eod_result.total_pnl:,.2f}", eod_result.passed),
        ("EOD drawdown (equity < floor)", f"Must stay above floor",
         f"Floor=${eod_result.final_floor:,.0f}, Min equity=${min(s.equity for s in eod_result.snapshots):,.0f}",
         not eod_result.breached),
    ]

    for name, required, actual, passed in criteria:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {name}")
        lines.append(f"         Required: {required}")
        lines.append(f"         Actual:   {actual}")

    lines += [
        "",
        "─── EOD Drawdown Floor Details ───",
        f"  Initial floor:              ${start_balance - eod_result.max_drawdown_limit:,.0f}  (start - max_drawdown)",
        f"  Final floor:                ${eod_result.final_floor:,.0f}",
        f"  Floor locked:               {eod_result.floor_locked}",
        f"  Lock threshold:             ${start_balance + eod_result.max_drawdown_limit:,.0f}  (start + max_drawdown)",
    ]
    if eod_result.floor_locked:
        lock_day = next((s.day for s in eod_result.snapshots if s.floor_locked), None)
        lines.append(f"  Floor locked on day:        {lock_day}")

    lines += [
        "",
        "─── Detailed Metrics ───",
        f"  Starting balance:           ${start_balance:,.2f}",
        f"  Final account equity:       ${eod_result.final_equity:,.2f}",
        f"  Peak equity:                ${eod_result.peak_equity:,.2f}",
        f"  Total PnL (funded phase):   ${eod_result.total_pnl:,.2f}",
        f"  Max trailing drawdown:      ${eod_result.max_trailing_dd:,.2f}",
        f"  Max drawdown parameter:     ${eod_result.max_drawdown_limit:,.0f}",
        f"  Profit target:              ${eod_result.profit_target:,.0f}",
        "",
        "─── Deterministic Run ───",
        f"  Result:                     {eod_result.reason}",
        f"  Passed:                     {eod_result.passed}",
        f"  Breached:                   {eod_result.breached}",
        f"  Days traded (funded):       {eod_result.days_traded}",
        f"  Day profit target reached:  {eod_result.day_profit_target_reached}"
        + (f"  ({daily_pnl.index[eod_result.day_profit_target_reached - 1]})"
           if eod_result.day_profit_target_reached and daily_pnl is not None
           and eod_result.day_profit_target_reached <= len(daily_pnl) else ""),
        f"  Day drawdown breached:      {eod_result.day_breached}"
        + (f"  ({daily_pnl.index[eod_result.day_breached - 1]})"
           if eod_result.day_breached and daily_pnl is not None
           and eod_result.day_breached <= len(daily_pnl) else ""),
        "",
        "─── Notes ───",
        "  - The funded phase uses the same EOD trailing drawdown rules.",
        "  - The consistency rule (40%) does NOT apply in the funded phase.",
        "  - The daily lock rule does NOT apply in the funded phase.",
        "  - Trades before the challenge pass date are excluded.",
        "",
    ]

    # ── Monte Carlo section ────────────────────────────────────────────
    if mc_result is not None:
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
        ]

        # Sequence risk verdict
        if mc_result.pass_rate >= 50.0:
            lines.append("  Sequence risk: LOW (MC pass rate >= 50%)")
        elif mc_result.pass_rate >= 25.0:
            lines.append("  Sequence risk: MODERATE (MC pass rate 25-50%)")
        else:
            lines.append("  Sequence risk: HIGH (MC pass rate < 25%)")

    # ── Verdict ────────────────────────────────────────────────────────
    lines += ["", "─── Verdict ───"]

    if eod_result.passed:
        day = eod_result.day_profit_target_reached
        date_str = ""
        if day and daily_pnl is not None and day <= len(daily_pnl):
            date_str = f" ({daily_pnl.index[day - 1]})"
        lines.append(f"  OUTCOME: FUNDED PHASE PASSED  (day {day}{date_str})")
        lines.append(f"  Profit target of ${eod_result.profit_target:,.0f} reached.")
    elif eod_result.breached:
        day = eod_result.day_breached
        date_str = ""
        if day and daily_pnl is not None and day <= len(daily_pnl):
            date_str = f" ({daily_pnl.index[day - 1]})"
        lines.append(f"  OUTCOME: FUNDED PHASE BREACHED  (day {day}{date_str})")
        lines.append("  Account equity fell below the floor.")
    else:
        lines.append("  OUTCOME: FUNDED PHASE TIME EXPIRED")
        lines.append("  Neither profit target nor drawdown breach occurred.")

    return "\n".join(lines)


def _generate_plot(
    eod_result: EODResult,
    daily_pnl: pd.Series,
    save_path: Path,
) -> None:
    """
    Generate and save a two-panel plot for the funded phase:
      1. Equity curve with floor (EOD drawdown limit), peak, target lines.
      2. Daily PnL bar chart.
    """
    dates = [s.date for s in eod_result.snapshots]
    equities = [s.equity for s in eod_result.snapshots]
    peak = [s.peak_equity for s in eod_result.snapshots]
    floors = [s.floor for s in eod_result.snapshots]
    daily_pnls = [s.daily_pnl for s in eod_result.snapshots]
    floor_locked = eod_result.floor_locked

    # Prepend day 0 (starting point before any trades)
    x = list(range(1, len(dates) + 1))
    start_bal = eod_result.start_balance
    x_ext = [0] + x
    equities_ext = [start_bal] + equities
    peak_ext = [start_bal] + peak
    floor_day0 = start_bal - eod_result.max_drawdown_limit
    floors_ext = [floor_day0] + floors

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Trader Launch — Funded Phase", fontsize=14, fontweight="bold")

    x_labels = dates
    step_tick = max(1, len(x_ext) // 15)
    lock_threshold = start_bal + eod_result.max_drawdown_limit

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — Equity curve with floor
    # ═══════════════════════════════════════════════════════════════════════
    ax1.plot(x_ext, equities_ext, "b-", linewidth=1.8, label="Equity")
    ax1.plot(x_ext, peak_ext, "g--", linewidth=1.0, alpha=0.7, label="Peak equity")
    ax1.step(x_ext, floors_ext, "r-", linewidth=2.0, where="pre", label="Floor (EOD drawdown limit)")
    ax1.fill_between(x_ext, floors_ext, min(floors_ext) - 500, alpha=0.08, color="red")

    ax1.axhline(start_bal, color="gray", linestyle=":", alpha=0.6, label=f"Start balance (${start_bal:,.0f})")
    ax1.axhline(
        start_bal + eod_result.profit_target,
        color="green", linestyle="--", alpha=0.5, label=f"Target (${eod_result.profit_target:,.0f})",
    )
    ax1.axhline(lock_threshold, color="orange", linestyle="-.", alpha=0.4,
                label=f"Lock threshold (${lock_threshold:,.0f})")

    if floor_locked:
        lock_day = next((i + 1 for i, s in enumerate(eod_result.snapshots) if s.floor_locked), None)
        if lock_day:
            ax1.annotate(
                "Floor locked",
                xy=(lock_day, start_bal),
                xytext=(lock_day + 0.5, start_bal + 300),
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
    ax1.set_title(f"Equity Curve with EOD Floor (starts at ${start_bal:,.0f})")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Daily PnL
    # ═══════════════════════════════════════════════════════════════════════
    colors = ["green" if v >= 0 else "red" for v in daily_pnls]
    ax2.bar(x, daily_pnls, color=colors, width=0.7, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Daily PnL ($)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Daily PnL Breakdown")

    all_labels = ["Start"] + list(x_labels)
    tick_positions = x_ext[::step_tick]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(
        [all_labels[i] for i in range(0, len(x_ext), step_tick)],
        rotation=45, ha="right", fontsize=8,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    trades: pd.DataFrame,
    challenge_result: Optional[dict] = None,
    config: Optional[dict] = None,
    reports_dir: Optional[Union[str, Path]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    n_mc_simulations: int = 5_000,
    mc_seed: Optional[int] = None,
) -> dict:
    """
    Run the Trader Launch Funded Phase analysis.

    Parameters
    ----------
    trades : pd.DataFrame
        Full trade data with a DatetimeIndex and ``pnl`` column.
    challenge_result : dict or None
        Result dict from ``Challenge_phase.run()``.
        Required to determine the challenge pass date and starting balance.
        If None, the funded phase cannot run.
    config : dict or None
        Optional rule overrides.  If None, rules are loaded from the YAML
        metadata file.
    reports_dir : str or Path or None
        Directory for text reports.  If None, no report is written.
    figures_dir : str or Path or None
        Directory for plots.  If None, no plot is generated.

    Returns
    -------
    dict with keys:
        deterministic_run  — EODResult (dataclass)
        daily_pnl          — pd.Series of daily PnL (funded phase only)
        summary            — plain dict for downstream aggregation
    """
    log.info("─" * 55)
    log.info("  Trader Launch — Funded Phase Calculation")
    log.info("─" * 55)

    if challenge_result is None:
        log.warning("  No challenge result provided — funded phase cannot run.")
        return {
            "deterministic_run": None,
            "daily_pnl": None,
            "summary": {"status": "skipped", "reason": "no_challenge_result"},
        }

    # ---- Unpack challenge result ------------------------------------------
    challenge_eod = challenge_result.get("deterministic_run")
    if challenge_eod is None or not challenge_eod.passed:
        log.warning("  Challenge phase was NOT passed — funded phase cannot run.")
        return {
            "deterministic_run": None,
            "daily_pnl": None,
            "summary": {"status": "skipped", "reason": "challenge_not_passed"},
        }

    challenge_passed_day = challenge_eod.day_profit_target_reached
    challenge_passed_date = challenge_result.get("daily_pnl", pd.Series()).index[
        challenge_passed_day - 1
    ]
    start_balance = 100_000.0

    log.info("  Challenge passed on day %d (%s)", challenge_passed_day, challenge_passed_date)
    log.info("  Starting balance: $%.2f  (reset to $100k for funded phase)", start_balance)

    # ---- Load rules -------------------------------------------------------
    rules = _load_rules() if config is None else config
    funded_cfg = rules["phases"]["funded"]

    max_drawdown = float(funded_cfg["max_total_drawdown"])
    profit_target = float(funded_cfg["profit_target"])

    # -----------------------------------------------------------------------
    # 1) Filter trades to funded phase only (after challenge pass date)
    # -----------------------------------------------------------------------
    funded_start = (pd.Timestamp(challenge_passed_date) + pd.Timedelta(days=1)).tz_localize("UTC")
    funded_trades = trades[trades.index.normalize() >= funded_start].copy()

    log.info("  Funded phase starts: %s", funded_start.date())
    log.info("  Trades in funded phase: %d / %d", len(funded_trades), len(trades))

    if len(funded_trades) == 0:
        log.warning("  No trades in funded phase — nothing to evaluate.")
        return {
            "deterministic_run": None,
            "daily_pnl": None,
            "summary": {"status": "skipped", "reason": "no_funded_trades"},
        }

    # -----------------------------------------------------------------------
    # 2) Build daily PnL (funded phase only)
    # -----------------------------------------------------------------------
    daily_pnl = aggregate_daily_pnl(funded_trades)
    log.info("  Unique trading days: %d", len(daily_pnl))
    log.info("  Total PnL: ${:,.2f}".format(daily_pnl.sum()))

    # -----------------------------------------------------------------------
    # 3) Deterministic EOD run (no consistency rule, no daily lock rule)
    # -----------------------------------------------------------------------
    eod_result = simulate(
        trades=funded_trades,
        start_balance=start_balance,
        max_drawdown=max_drawdown,
        profit_target=profit_target,
    )

    log.info("  Deterministic run: %s", eod_result.reason)
    if eod_result.passed:
        log.info("    -> PASSED  (day %d, max DD $%.2f)", eod_result.day_profit_target_reached, eod_result.max_trailing_dd)
    elif eod_result.breached:
        log.info("    -> BREACH  (day %d, max DD $%.2f)", eod_result.day_breached, eod_result.max_trailing_dd)
    else:
        log.info("    -> TIME EXPIRED  (max DD $%.2f)", eod_result.max_trailing_dd)

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
        seed=mc_seed,
    )
    log.info("  MC pass rate: %.1f%%  (fail: %.1f%%)", mc_result.pass_rate, mc_result.fail_rate)

    # -----------------------------------------------------------------------
    # 5) Write text report
    # -----------------------------------------------------------------------
    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        report_path = reports_path / "Funded_phase.txt"

        report_text = _format_result_text(
            eod_result=eod_result,
            mc_result=mc_result,
            funded_cfg=funded_cfg,
            start_balance=start_balance,
            challenge_passed_date=str(challenge_passed_date),
            total_pnl_in_funded=eod_result.total_pnl,
            days_traded_in_funded=eod_result.days_traded,
            daily_pnl=daily_pnl,
        )
        report_path.write_text(report_text, encoding="utf-8")
        log.info("  Report written to %s", report_path)

    # -----------------------------------------------------------------------
    # 5) Generate plot
    # -----------------------------------------------------------------------
    if figures_dir is not None:
        figures_path = Path(figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        plot_path = figures_path / "Funded_phase_calculation_plot.png"
        _generate_plot(eod_result, daily_pnl, plot_path)

    # -----------------------------------------------------------------------
    # Return aggregated results
    # -----------------------------------------------------------------------
    return {
        "deterministic_run": eod_result,
        "monte_carlo": mc_result,
        "daily_pnl": daily_pnl,
        "summary": {
            "status": "completed",
            "funded_passed": eod_result.passed,
            "funded_reason": eod_result.reason,
            "funded_days": eod_result.days_traded,
            "funded_day_target": eod_result.day_profit_target_reached,
            "funded_max_dd": round(eod_result.max_trailing_dd, 2),
            "funded_total_pnl": round(eod_result.total_pnl, 2),
            "mc_pass_rate_pct": round(mc_result.pass_rate, 2),
            "mc_fail_rate_pct": round(mc_result.fail_rate, 2),
            "mc_n_simulations": mc_result.n_simulations,
        },
    }
