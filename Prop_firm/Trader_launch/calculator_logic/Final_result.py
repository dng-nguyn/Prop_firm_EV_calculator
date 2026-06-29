"""
Trader Launch — Final Result Calculation
=========================================

Combines Challenge, Funded, and Live phase outputs into a final summary.

Key computations
----------------
- **Expected attempts to pass challenge**   — 1 / challenge_mc_pass_rate
- **Expected attempts to first payout**     — 1 / (challenge_rate × funded_rate)
- **Expected cost before first payout**     — eval_fee × attempts_to_first_payout
- **Expected value per attempt**            — (trader_profit × prob_reaching_live) − eval_fee

Generates ``Final_result.txt`` and ``Final_result_metadata.yaml``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

log = logging.getLogger("TraderLaunch.FinalResult")


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
# Public API
# ---------------------------------------------------------------------------


def run(
    challenge_result: Optional[dict] = None,
    funded_result: Optional[dict] = None,
    live_result: Optional[dict] = None,
    reports_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Combine all phase results into a final summary.

    Parameters
    ----------
    challenge_result : dict or None
        Output of ``Challenge_phase.run()``.
    funded_result : dict or None
        Output of ``Funded_phase.run()``.
    live_result : dict or None
        Output of ``Live_phase.run()``.
    reports_dir : str or Path or None
        Directory for text / YAML reports.

    Returns
    -------
    dict with keys:
        summary  — plain dict for downstream aggregation
    """
    log.info("─" * 55)
    log.info("  Trader Launch — Final Result Aggregation")
    log.info("─" * 55)

    # ---- Extract metrics from each phase --------------------------------
    rules = _load_rules()
    eval_fee = float(rules.get("general", {}).get("evaluation_fee", 45))

    # Challenge MC pass rate
    challenge_rate = 0.0
    if challenge_result:
        challenge_rate = challenge_result.get("summary", {}).get("mc_pass_rate_pct", 0.0)
    log.info("  Challenge MC pass rate:   %.2f%%", challenge_rate)

    # Funded MC pass rate
    funded_rate = 0.0
    if funded_result:
        funded_rate = funded_result.get("summary", {}).get("mc_pass_rate_pct", 0.0)
    log.info("  Funded MC pass rate:      %.2f%%", funded_rate)

    # Live trader profit
    live_summary = live_result.get("summary", {}) if live_result else {}
    trader_profit = float(live_summary.get("total_trader_profit", 0.0))
    total_withdrawn = float(live_summary.get("total_withdrawn", 0.0))
    account_status = live_summary.get("account_status", "no_data")
    log.info("  Live trader profit:       $%.2f", trader_profit)

    # ---- Extract MC day stats -------------------------------------------
    challenge_mc = challenge_result.get("monte_carlo") if challenge_result else None
    funded_mc = funded_result.get("monte_carlo") if funded_result else None

    challenge_avg_pass_days = getattr(challenge_mc, "avg_pass_days", None) if challenge_mc else None
    challenge_avg_fail_days = getattr(challenge_mc, "avg_fail_days", None) if challenge_mc else None
    funded_avg_pass_days = getattr(funded_mc, "avg_pass_days", None) if funded_mc else None
    funded_avg_fail_days = getattr(funded_mc, "avg_fail_days", None) if funded_mc else None

    # ---- Compute statistics ----------------------------------------------
    challenge_rate_dec = challenge_rate / 100.0
    funded_rate_dec = funded_rate / 100.0

    # Expected attempts to pass challenge
    exp_attempts_challenge = 1.0 / challenge_rate_dec if challenge_rate_dec > 0 else float("inf")
    exp_fails_challenge = exp_attempts_challenge - 1.0 if exp_attempts_challenge != float("inf") else float("inf")
    exp_cost_before_challenge = eval_fee * exp_attempts_challenge if exp_attempts_challenge != float("inf") else float("inf")

    # Probability of reaching live phase
    prob_reaching_live = challenge_rate_dec * funded_rate_dec

    # Expected attempts to get first payout
    exp_attempts_to_payout = 1.0 / prob_reaching_live if prob_reaching_live > 0 else float("inf")
    exp_cost_to_first_payout = eval_fee * exp_attempts_to_payout if exp_attempts_to_payout != float("inf") else float("inf")

    # Expected trading days to first payout
    exp_days_challenge = None
    if challenge_avg_pass_days is not None and challenge_avg_fail_days is not None and exp_fails_challenge != float("inf"):
        exp_days_challenge = (exp_fails_challenge * challenge_avg_fail_days) + challenge_avg_pass_days

    exp_days_funded = funded_avg_pass_days  # only counting the successful pass

    exp_days_live_to_first_wd = None
    live_obj = live_result.get("live_result") if live_result else None
    if live_obj and hasattr(live_obj, "withdrawals") and live_obj.withdrawals:
        exp_days_live_to_first_wd = float(live_obj.withdrawals[0].day)

    exp_total_days_to_payout = None
    if exp_days_challenge is not None and exp_days_funded is not None and exp_days_live_to_first_wd is not None:
        exp_total_days_to_payout = exp_days_challenge + exp_days_funded + exp_days_live_to_first_wd

    # Expected value per attempt
    ev_per_attempt = (trader_profit * prob_reaching_live) - eval_fee

    # Expected value per pipeline (one full pass through all phases)
    total_cost = eval_fee * exp_attempts_to_payout if exp_attempts_to_payout != float("inf") else float("inf")
    ev_per_pipeline = trader_profit - total_cost if total_cost != float("inf") else float("-inf")

    # ---- Log summary ----------------------------------------------------
    log.info("  Eval fee:                $%.2f", eval_fee)
    log.info("  Prob(reaching live):     %.2f%%", prob_reaching_live * 100)
    log.info("  Exp attempts to payout:  %.2f", exp_attempts_to_payout)
    log.info("  Exp cost to 1st payout:  $%.2f", exp_cost_to_first_payout)
    if exp_total_days_to_payout is not None:
        log.info("  Exp days to 1st payout:  %.1f  (challenge: %.1f + funded: %.1f + live: %.0f)",
                 exp_total_days_to_payout, exp_days_challenge, exp_days_funded, exp_days_live_to_first_wd)
    log.info("  EV per attempt:          $%.2f", ev_per_attempt)
    log.info("  EV per pipeline:         $%.2f", ev_per_pipeline)

    # ---- Build summary dict ---------------------------------------------
    summary = {
        "firm": "Trader_launch",
        "eval_fee_usd": eval_fee,
        "challenge": {
            "mc_pass_rate_pct": round(challenge_rate, 2),
            "expected_attempts": round(exp_attempts_challenge, 2) if exp_attempts_challenge != float("inf") else None,
            "expected_cost_before_pass_usd": round(exp_cost_before_challenge, 2) if exp_cost_before_challenge != float("inf") else None,
        },
        "funded": {
            "mc_pass_rate_pct": round(funded_rate, 2),
        },
        "live": {
            "account_status": account_status,
            "total_withdrawn_usd": round(total_withdrawn, 2),
            "trader_profit_usd": round(trader_profit, 2),
        },
        "final": {
            "prob_reaching_live_pct": round(prob_reaching_live * 100, 2),
            "expected_attempts_to_first_payout": round(exp_attempts_to_payout, 2) if exp_attempts_to_payout != float("inf") else None,
            "expected_cost_to_first_payout_usd": round(exp_cost_to_first_payout, 2) if exp_cost_to_first_payout != float("inf") else None,
            "expected_days_to_first_payout": round(exp_total_days_to_payout, 1) if exp_total_days_to_payout is not None else None,
            "expected_days_breakdown": {
                "challenge_days": round(exp_days_challenge, 1) if exp_days_challenge is not None else None,
                "funded_days": round(exp_days_funded, 1) if exp_days_funded is not None else None,
                "live_days_to_first_withdrawal": round(exp_days_live_to_first_wd, 0) if exp_days_live_to_first_wd is not None else None,
            } if exp_total_days_to_payout is not None else None,
            "expected_value_per_attempt_usd": round(ev_per_attempt, 2),
            "expected_value_per_pipeline_usd": round(ev_per_pipeline, 2) if ev_per_pipeline != float("-inf") else None,
        },
    }

    # ---- Write reports --------------------------------------------------
    if reports_dir is not None:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)

        # Text report
        report_path = reports_path / "Final_result.txt"
        report_text = _format_result_text(
            summary, eval_fee, challenge_rate, funded_rate,
            trader_profit, prob_reaching_live, exp_attempts_to_payout,
            exp_cost_to_first_payout, ev_per_attempt, ev_per_pipeline,
            exp_total_days_to_payout, exp_days_challenge, exp_days_funded,
            exp_days_live_to_first_wd,
        )
        report_path.write_text(report_text, encoding="utf-8")
        log.info("  Report written to %s", report_path)

        # YAML metadata
        yaml_path = reports_path / "Final_result_metadata.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(summary, fh, default_flow_style=False, sort_keys=False)
        log.info("  Metadata written to %s", yaml_path)

    return {"summary": summary}


def _format_result_text(
    summary: dict,
    eval_fee: float,
    challenge_rate: float,
    funded_rate: float,
    trader_profit: float,
    prob_reaching_live: float,
    exp_attempts_to_payout: float,
    exp_cost_to_first_payout: float,
    ev_per_attempt: float,
    ev_per_pipeline: float,
    exp_total_days_to_payout: Optional[float] = None,
    exp_days_challenge: Optional[float] = None,
    exp_days_funded: Optional[float] = None,
    exp_days_live_to_first_wd: Optional[float] = None,
) -> str:
    """Build the human-readable Final_result.txt content."""
    def _inf(val):
        return "N/A" if val is None or val == float("inf") or val == float("-inf") else f"{val:.2f}"

    lines = [
        "# Trader Launch — Final Results",
        "# ==============================",
        "",
        "─── Configuration ───",
        f"  Evaluation fee:             ${eval_fee:,.0f}",
        "",
        "─── Phase Pass Rates (Monte Carlo) ───",
        f"  Challenge phase:            {challenge_rate:.2f}%",
        f"  Funded phase:               {funded_rate:.2f}%",
        f"  Combined (→ live):          {prob_reaching_live*100:.2f}%",
        "",
        "─── Live Phase Results ───",
        f"  Account status:             {summary['live']['account_status']}",
        f"  Total withdrawn:            ${summary['live']['total_withdrawn_usd']:,.2f}",
        f"  Trader profit (55%):        ${trader_profit:,.2f}",
        "",
        "─── Expected Value ───",
        f"  Expected attempts to first payout:   {_inf(exp_attempts_to_payout)}",
        f"  Expected cost to first payout:       ${_inf(exp_cost_to_first_payout)}",
        "",
        f"  Expected value per attempt:           ${ev_per_attempt:,.2f}",
    ]

    if ev_per_pipeline != float("-inf"):
        lines.append(f"  Expected value per pipeline:        ${ev_per_pipeline:,.2f}")
    else:
        lines.append("  Expected value per pipeline:        N/A")

    if exp_total_days_to_payout is not None:
        lines += [
            "",
            "─── Expected Trading Days to First Payout ───",
            f"  Challenge phase (incl. retries):  {exp_days_challenge:.1f} days",
            f"  Funded phase:                      {exp_days_funded:.1f} days",
            f"  Live phase to first withdrawal:    {exp_days_live_to_first_wd:.0f} days",
            f"  ───────────────────────────────────────────",
            f"  Total:                             {exp_total_days_to_payout:.1f} days",
        ]

    lines += [
        "",
        "─── Interpretation ───",
    ]

    if ev_per_attempt > 0:
        lines.append("  ✅  Positive expected value — the pipeline is profitable on average.")
    elif ev_per_attempt < 0:
        lines.append("  ⚠️   Negative expected value — the pipeline loses money on average.")
    else:
        lines.append("  ➖  Expected value is zero — break-even.")

    if prob_reaching_live > 0.5:
        lines.append("  📈  High probability of reaching the live phase.")
    elif prob_reaching_live > 0.2:
        lines.append("  📊  Moderate probability of reaching the live phase.")
    else:
        lines.append("  📉  Low probability of reaching the live phase.")

    return "\n".join(lines)
