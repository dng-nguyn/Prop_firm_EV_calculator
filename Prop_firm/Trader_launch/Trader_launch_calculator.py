"""
Trader Launch Calculator — Entry point for Trader Launch prop firm evaluation.
Runs the Challenge and Funded phase calculations, then produces final results.

Execution sequence:
  1. Trader_launch_rules_filter_Challenge_phase.py  — Validate trades against challenge rules
  2. Challenge_phase.py                              — Compute challenge phase metrics
  3. Trader_launch_rules_filter_Funded_phase.py      — Validate trades against funded rules
  4. Funded_phase.py                                 — Compute funded phase metrics
  5. Live_phase.py                                   — Simulate live funded trading
  6. Final_result.py                                 — Combine results and generate reports
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # Prop_firm_calculator/
FIRM_DIR     = Path(__file__).resolve().parent               # Trader_launch/
LOGIC_DIR    = FIRM_DIR / "calculator_logic"
REPORTS_DIR  = FIRM_DIR / "reports"
RESULTS_DIR  = REPORTS_DIR / "results"
FIGURES_DIR  = REPORTS_DIR / "figures_png"
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRADES_CSV   = RAW_DATA_DIR / "trades.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TraderLaunch")


# ---------------------------------------------------------------------------
# Imports from child calculator_logic modules
# ---------------------------------------------------------------------------
# Each module is expected to expose a "run()" function that accepts the
# relevant data / configuration and returns results.

try:
    from calculator_logic.Trader_launch_rules_filter_Challenge_phase import (
        run as run_challenge_rules_filter,
    )
except ImportError:
    log.warning("Trader_launch_rules_filter_Challenge_phase not yet implemented — skipping import.")
    run_challenge_rules_filter = None

try:
    from calculator_logic.Challenge_phase import (
        run as run_challenge_phase,
    )
except ImportError:
    log.warning("Challenge_phase not yet implemented — skipping import.")
    run_challenge_phase = None

try:
    from calculator_logic.Trader_launch_rules_filter_Funded_phase import (
        run as run_funded_rules_filter,
    )
except ImportError:
    log.warning("Trader_launch_rules_filter_Funded_phase not yet implemented — skipping import.")
    run_funded_rules_filter = None

try:
    from calculator_logic.Funded_phase import (
        run as run_funded_phase,
    )
except ImportError:
    log.warning("Funded_phase not yet implemented — skipping import.")
    run_funded_phase = None

try:
    from calculator_logic.Live_phase import (
        run as run_live_phase,
    )
except ImportError:
    log.warning("Live_phase not yet implemented — skipping import.")
    run_live_phase = None

try:
    from calculator_logic.Final_result import (
        run as run_final_result,
    )
except ImportError:
    log.warning("Final_result not yet implemented — skipping import.")
    run_final_result = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for d in [RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_trades(csv_path: str | Path = TRADES_CSV) -> pd.DataFrame:
    """
    Load trade data from CSV.

    Expected columns:
        entry_time, exit_time, side, entry_price, exit_price,
        quantity, pnl, duration_min

    Returns
    -------
    pd.DataFrame with datetime index derived from entry_time.
    """
    log.info("Loading trades from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])
    df.set_index("entry_time", inplace=True)

    # Ensure index is a proper DatetimeIndex. Timezone-aware timestamps
    # may be stored as object-dtype Timestamp objects, so convert via
    # to_datetime with utc=True to handle them robustly.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    log.info("  → %d trades loaded  (index: %s)", len(df), type(df.index).__name__)
    return df


def step_runner(step_num: int, step_name: str, step_func, *args, **kwargs):
    """
    Execute a single pipeline step with error handling.

    Parameters
    ----------
    step_num   : int    — position in the pipeline (1‑based)
    step_name  : str    — human-readable label
    step_func  : callable or None — the module's run function
    *args, **kwargs     — forwarded to step_func

    Returns
    -------
    The result of step_func, or None if the module is unavailable.
    """
    log.info("═" * 60)
    log.info("STEP %d / 6 — %s", step_num, step_name)
    log.info("═" * 60)

    if step_func is None:
        log.warning("  ⚠  Module not available — step skipped.")
        return None

    try:
        result = step_func(*args, **kwargs)
        log.info("  ✔  %s completed successfully.", step_name)
        return result
    except Exception as exc:
        log.error("  ✘  %s FAILED — %s", step_name, exc)
        raise


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    trades: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict:
    """
    Execute the full 6‑step Trader Launch evaluation pipeline.

    Parameters
    ----------
    trades : pd.DataFrame or None
        Pre-loaded trade data.  If None, loads from the default CSV.
    config : dict or None
        Optional configuration overrides (e.g. account size, rule variants).

    Returns
    -------
    dict with keys:
        "challenge_rules_result"  — output of step 1
        "challenge_result"        — output of step 2
        "funded_rules_result"     — output of step 3
        "funded_result"           — output of step 4
        "live_result"             — output of step 5
        "final_result"            — output of step 6
    """
    ensure_directories()

    # ---- Load data --------------------------------------------------------
    if trades is None:
        trades = load_trades()

    results: dict = {}

    # ---- Step 1: Challenge Phase — Rules Filter ---------------------------
    results["challenge_rules_result"] = step_runner(
        1,
        "Challenge Phase — Rules Filter",
        run_challenge_rules_filter,
        trades=trades,
        config=config,
    )

    # ---- Step 2: Challenge Phase — Calculation ----------------------------
    results["challenge_result"] = step_runner(
        2,
        "Challenge Phase — Calculation",
        run_challenge_phase,
        trades=trades,
        config=config,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # ---- Step 3: Funded Phase — Rules Filter ------------------------------
    results["funded_rules_result"] = step_runner(
        3,
        "Funded Phase — Rules Filter",
        run_funded_rules_filter,
        trades=trades,
        config=config,
    )

    # ---- Step 4: Funded Phase — Calculation -------------------------------
    results["funded_result"] = step_runner(
        4,
        "Funded Phase — Calculation",
        run_funded_phase,
        trades=trades,
        challenge_result=results.get("challenge_result"),
        config=config,
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # ---- Step 5: Live Funded Phase ---------------------------------------
    results["live_result"] = step_runner(
        5,
        "Live Funded Phase — Simulation",
        run_live_phase,
        trades=trades,
        funded_result=results.get("funded_result"),
        reports_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    )

    # ---- Step 6: Final Result ---------------------------------------------
    results["final_result"] = step_runner(
        6,
        "Final Result — Aggregation & Reports",
        run_final_result,
        challenge_result=results.get("challenge_result"),
        funded_result=results.get("funded_result"),
        live_result=results.get("live_result"),
        reports_dir=RESULTS_DIR,
    )

    log.info("═" * 60)
    log.info("Trader Launch pipeline complete.")
    log.info("═" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """CLI entry point — load trades and run the full pipeline."""
    print()
    log.info("🚀  Trader Launch Calculator — starting evaluation pipeline")
    print()

    trades = load_trades()
    run_pipeline(trades=trades)


if __name__ == "__main__":
    sys.exit(main())
