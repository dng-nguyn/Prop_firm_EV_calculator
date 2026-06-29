"""
Trader Launch — Live (Funded Trading) Phase
============================================

Simulates the live funded account after the evaluation phases are passed.

Rules
-----
- Starting balance: $101,000 ($100k start + $1k funded profit target)
- Trading begins the day **after** the funded phase is passed
- No EOD drawdown / floor rules — just track daily balance
- **Profit withdrawal**: if end-of-day balance exceeds $101,200,
  the excess above $101,000 is withdrawn. Trader receives **55 %**.
- **Termination**: if balance ever drops to ≤ $100,000, the account
  is terminated immediately.
- Runs until termination or the end of available trade data.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from Common.EOD.trailing_drawdown import aggregate_daily_pnl

log = logging.getLogger("TraderLaunch.Live")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
START_BALANCE = 101_000.0
TERMINATION_THRESHOLD = 100_000.0
WITHDRAWAL_THRESHOLD = 101_200.0
BUFFER = 101_000.0
PROFIT_SPLIT = 0.55  # trader's share


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Withdrawal:
    day: int
    date: str
    balance_before: float
    withdrawn: float
    trader_share: float
    balance_after: float


@dataclass
class LiveResult:
    """Result of the live phase simulation."""

    status: str                     # "active" | "terminated" | "no_data"
    start_balance: float
    final_balance: float
    days_traded: int
    total_withdrawn: float
    total_trader_profit: float
    n_withdrawals: int
    termination_day: Optional[int]
    termination_date: Optional[str]
    withdrawals: list[Withdrawal] = field(repr=False, default_factory=list)
    daily_balances: list[float] = field(repr=False, default_factory=list)
    daily_dates: list[str] = field(repr=False, default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_live_phase(daily_pnl: pd.Series) -> LiveResult:
    """
    Walk through daily PnL values and simulate the live account.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily net PnL values indexed by date, in chronological order.

    Returns
    -------
    LiveResult
    """
    if len(daily_pnl) == 0:
        return LiveResult(
            status="no_data",
            start_balance=START_BALANCE,
            final_balance=START_BALANCE,
            days_traded=0,
            total_withdrawn=0.0,
            total_trader_profit=0.0,
            n_withdrawals=0,
            termination_day=None,
            termination_date=None,
        )

    balance = START_BALANCE
    total_withdrawn = 0.0
    total_trader_profit = 0.0
    withdrawals: list[Withdrawal] = []
    daily_balances: list[float] = [balance]
    daily_dates: list[str] = [f"Start ({daily_pnl.index[0]})"]

    status = "active"
    termination_day: Optional[int] = None
    termination_date: Optional[str] = None

    for day_idx in range(len(daily_pnl)):
        day_pnl = float(daily_pnl.iloc[day_idx])
        day_date = str(daily_pnl.index[day_idx])
        balance += day_pnl

        # ---- Termination check -------------------------------------------
        if balance <= TERMINATION_THRESHOLD:
            status = "terminated"
            termination_day = day_idx + 1
            termination_date = day_date
            daily_balances.append(balance)
            daily_dates.append(day_date)
            log.info(
                "    Day %d (%s): balance $%.2f → TERMINATED (≤ $%.0f)",
                day_idx + 1, day_date, balance, TERMINATION_THRESHOLD,
            )
            break

        # ---- Withdrawal check --------------------------------------------
        if balance > WITHDRAWAL_THRESHOLD:
            withdraw = balance - BUFFER
            trader_share = round(withdraw * PROFIT_SPLIT, 2)
            total_withdrawn += withdraw
            total_trader_profit += trader_share

            withdrawals.append(Withdrawal(
                day=day_idx + 1,
                date=day_date,
                balance_before=balance,
                withdrawn=round(withdraw, 2),
                trader_share=trader_share,
                balance_after=BUFFER,
            ))

            log.info(
                "    Day %d (%s): balance $%.2f → withdraw $%.2f "
                "(trader gets $%.2f) → reset to $%.0f",
                day_idx + 1, day_date, balance,
                withdraw, trader_share, BUFFER,
            )

            balance = BUFFER  # reset

        daily_balances.append(balance)
        daily_dates.append(day_date)

    return LiveResult(
        status=status,
        start_balance=START_BALANCE,
        final_balance=balance,
        days_traded=len(daily_pnl) if status != "terminated" else termination_day or 0,
        total_withdrawn=round(total_withdrawn, 2),
        total_trader_profit=round(total_trader_profit, 2),
        n_withdrawals=len(withdrawals),
        termination_day=termination_day,
        termination_date=termination_date,
        withdrawals=withdrawals,
        daily_balances=daily_balances,
        daily_dates=daily_dates,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _generate_plot(result: LiveResult, save_path: Path) -> None:
    """Generate a 2-panel plot: equity curve + withdrawal markers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Trader Launch — Live Funded Phase", fontsize=14, fontweight="bold")

    x = list(range(len(result.daily_balances)))
    x_labels = result.daily_dates
    step = max(1, len(x) // 15)

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — Balance curve
    # ═══════════════════════════════════════════════════════════════════════
    ax1.plot(x, result.daily_balances, "b-", linewidth=1.8, label="Account balance")

    # Start line
    ax1.axhline(START_BALANCE, color="gray", linestyle=":", alpha=0.6,
                label=f"Start / buffer (${START_BALANCE:,.0f})")
    # Termination line
    ax1.axhline(TERMINATION_THRESHOLD, color="red", linestyle="--", alpha=0.6,
                label=f"Termination (${TERMINATION_THRESHOLD:,.0f})")
    # Withdrawal threshold
    ax1.axhline(WITHDRAWAL_THRESHOLD, color="orange", linestyle="-.", alpha=0.4,
                label=f"Withdrawal trigger (${WITHDRAWAL_THRESHOLD:,.0f})")

    # Mark withdrawals
    for w in result.withdrawals:
        # The withdrawal happens at the point after the day's PnL, before reset
        w_x = w.day  # day index (1-based) = position in daily_balances
        ax1.scatter([w_x], [w.balance_before], color="green", s=60, zorder=5)
        ax1.annotate(
            f"${w.withdrawn:,.0f}",
            xy=(w_x, w.balance_before),
            xytext=(5, 10), textcoords="offset points",
            fontsize=7, color="green", fontweight="bold",
        )

    # Mark termination
    if result.termination_day:
        ax1.scatter([result.termination_day], [result.final_balance],
                    color="red", s=80, zorder=5, marker="x")
        ax1.annotate(
            "TERMINATED",
            xy=(result.termination_day, result.final_balance),
            xytext=(10, -20), textcoords="offset points",
            fontsize=9, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax1.set_ylabel("Account Balance ($)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Account Balance  —  Total withdrawn: ${result.total_withdrawn:,.2f}  |  "
                  f"Trader profit: ${result.total_trader_profit:,.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Withdrawal bars
    # ═══════════════════════════════════════════════════════════════════════
    if result.withdrawals:
        w_days = [w.day for w in result.withdrawals]
        w_vals = [w.withdrawn for w in result.withdrawals]
        w_trader = [w.trader_share for w in result.withdrawals]
        ax2.bar(w_days, w_vals, width=0.6, alpha=0.7, color="steelblue", label="Total withdrawal")
        ax2.bar(w_days, w_trader, width=0.6, alpha=0.9, color="green", label="Trader share (55%)")
        ax2.set_ylabel("Withdrawal Amount ($)")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.set_title(f"Withdrawals — {result.n_withdrawals} event(s), "
                      f"trader total: ${result.total_trader_profit:,.2f}")
    else:
        ax2.text(0.5, 0.5, "No withdrawals occurred",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Withdrawals — None")

    ax2.set_xlabel("Trading Day")
    ax2.grid(True, alpha=0.3)

    tick_positions = x[::step]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x), step)],
                        rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _format_result_text(result: LiveResult) -> str:
    """Build the human-readable Live_phase.txt content."""
    lines = [
        "# Trader Launch — Live Funded Phase Results",
        "# ===========================================",
        "",
        "─── Live Phase Configuration ───",
        f"  Starting balance:           ${START_BALANCE:,.0f}",
        f"  Withdrawal threshold:        ${WITHDRAWAL_THRESHOLD:,.0f} (keep ${BUFFER:,.0f} buffer)",
        f"  Termination threshold:       ${TERMINATION_THRESHOLD:,.0f}",
        f"  Profit split (trader):       {PROFIT_SPLIT:.0%}",
        "",
        "─── Results ───",
        f"  Account status:             {result.status}",
        f"  Days traded:                {result.days_traded}",
        f"  Final balance:              ${result.final_balance:,.2f}",
        f"  Total withdrawn:            ${result.total_withdrawn:,.2f}",
        f"  Trader profit (55%):        ${result.total_trader_profit:,.2f}",
        f"  Number of withdrawals:      {result.n_withdrawals}",
    ]

    if result.termination_day:
        lines += [
            "",
            "─── Termination ───",
            f"  Terminated on day:          {result.termination_day}  ({result.termination_date})",
            f"  Final balance:              ${result.final_balance:,.2f}",
        ]

    if result.withdrawals:
        lines += [
            "",
            "─── Withdrawal History ───",
            "  Day  Date         Balance Before  Withdrawn  Trader Gets  Balance After",
            "  ───  ──────────  ─────────────  ─────────  ───────────  ─────────────",
        ]
        for w in result.withdrawals:
            lines.append(
                f"  {w.day:>3}  {w.date:10}  "
                f"${w.balance_before:>8,.2f}   "
                f"${w.withdrawn:>7,.2f}  "
                f"${w.trader_share:>9,.2f}  "
                f"${w.balance_after:>8,.2f}"
            )

    lines += [
        "",
        "─── Verdict ───",
    ]
    if result.status == "active":
        lines.append("  OUTCOME: LIVE ACCOUNT ACTIVE (end of data)")
        lines.append(f"  Final balance: ${result.final_balance:,.2f}")
    elif result.status == "terminated":
        lines.append(f"  OUTCOME: LIVE ACCOUNT TERMINATED on day {result.termination_day}")
        lines.append(f"  Balance dropped to ${result.final_balance:,.2f} (≤ ${TERMINATION_THRESHOLD:,.0f})")
    else:
        lines.append("  OUTCOME: NO DATA AVAILABLE")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    trades: pd.DataFrame,
    funded_result: Optional[dict] = None,
    reports_dir: Optional[Union[str, Path]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Run the Trader Launch Live Funded Phase analysis.

    Parameters
    ----------
    trades : pd.DataFrame
        Full trade data with a DatetimeIndex and ``pnl`` column.
    funded_result : dict or None
        Result dict from ``Funded_phase.run()``.
        Required to determine when the funded phase was passed.
        If None, the live phase starts from the beginning of trades.
    reports_dir : str or Path or None
        Directory for text reports.  If None, no report is written.
    figures_dir : str or Path or None
        Directory for plots.  If None, no plot is generated.

    Returns
    -------
    dict with keys:
        live_result  — LiveResult dataclass
        summary      — plain dict with key metrics for downstream aggregation
    """
    log.info("─" * 55)
    log.info("  Trader Launch — Live Funded Phase")
    log.info("─" * 55)

    # ---- Determine live phase start date ---------------------------------
    live_start = None
    if funded_result is not None:
        funded_eod = funded_result.get("deterministic_run")
        funded_summary = funded_result.get("summary", {})
        funded_passed_day = funded_summary.get("funded_day_target")
        funded_daily = funded_result.get("daily_pnl")

        if funded_eod is not None and funded_passed_day is not None and funded_daily is not None:
            if funded_passed_day <= len(funded_daily):
                funded_pass_date = funded_daily.index[funded_passed_day - 1]
                live_start = (pd.Timestamp(funded_pass_date) + pd.Timedelta(days=1)).tz_localize("UTC")
                log.info("  Funded passed on day %s (%s)",
                         funded_passed_day, funded_pass_date)
        elif funded_eod is not None and funded_daily is not None and len(funded_daily) > 0:
            # Funded failed — use last funded day as live start
            fallback_day = min(funded_eod.days_traded, len(funded_daily))
            funded_end_date = funded_daily.index[fallback_day - 1]
            live_start = (pd.Timestamp(funded_end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
            log.warning("  Funded NOT passed — using last funded day %d (%s) as live start",
                        fallback_day, funded_end_date)

    if live_start is not None:
        log.info("  Live phase starts: %s", live_start.date())
        live_trades = trades[trades.index.normalize() >= live_start].copy()
    else:
        log.info("  No funded result — starting live phase from beginning")
        live_trades = trades.copy()

    log.info("  Trades in live phase: %d / %d", len(live_trades), len(trades))

    if len(live_trades) == 0:
        log.warning("  No trades in live phase — nothing to evaluate.")
        return {
            "live_result": LiveResult(
                status="no_data",
                start_balance=START_BALANCE,
                final_balance=START_BALANCE,
                days_traded=0,
                total_withdrawn=0.0,
                total_trader_profit=0.0,
                n_withdrawals=0,
                termination_day=None,
                termination_date=None,
            ),
            "summary": {"status": "skipped", "reason": "no_live_trades"},
        }

    # -----------------------------------------------------------------------
    # 1) Aggregate daily PnL
    # -----------------------------------------------------------------------
    daily_pnl = aggregate_daily_pnl(live_trades)
    log.info("  Unique trading days: %d", len(daily_pnl))
    log.info("  Total PnL in live period: ${:,.2f}".format(daily_pnl.sum()))

    # -----------------------------------------------------------------------
    # 2) Simulate live phase
    # -----------------------------------------------------------------------
    result = simulate_live_phase(daily_pnl)

    log.info("  Live phase result: %s", result.status)
    log.info("    Days traded: %d", result.days_traded)
    log.info("    Final balance: $%.2f", result.final_balance)
    log.info("    Total withdrawn: $%.2f", result.total_withdrawn)
    log.info("    Trader profit: $%.2f", result.total_trader_profit)
    if result.n_withdrawals:
        log.info("    Withdrawals: %d", result.n_withdrawals)
    if result.termination_day:
        log.info("    Terminated on day %d", result.termination_day)

    # -----------------------------------------------------------------------
    # 3) Write text report
    # -----------------------------------------------------------------------
    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        report_path = reports_path / "Live_phase.txt"
        report_path.write_text(_format_result_text(result), encoding="utf-8")
        log.info("  Report written to %s", report_path)

    # -----------------------------------------------------------------------
    # 4) Generate plot
    # -----------------------------------------------------------------------
    if figures_dir is not None:
        figures_path = Path(figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        plot_path = figures_path / "Live_phase_plot.png"
        _generate_plot(result, plot_path)

    # -----------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------
    return {
        "live_result": result,
        "summary": {
            "status": "completed",
            "account_status": result.status,
            "days_traded": result.days_traded,
            "final_balance": round(result.final_balance, 2),
            "total_withdrawn": round(result.total_withdrawn, 2),
            "total_trader_profit": round(result.total_trader_profit, 2),
            "n_withdrawals": result.n_withdrawals,
            "termination_day": result.termination_day,
        },
    }
