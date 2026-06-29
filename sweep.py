"""Sweep block sizes and live phase buffer to find optimal EV."""
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/root/slop/Prop_firm_EV_calculator')

import yaml
import pandas as pd
from Prop_firm.Trader_launch.Trader_launch_calculator import (
    load_trades, TRADES_CSV, RESULTS_DIR, FIGURES_DIR, ensure_directories,
)
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Challenge_phase import run as run_cr
from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import run as run_ch
from Prop_firm.Trader_launch.calculator_logic.Trader_launch_rules_filter_Funded_phase import run as run_fr
from Prop_firm.Trader_launch.calculator_logic.Funded_phase import run as run_fu
from Prop_firm.Trader_launch.calculator_logic.Live_phase import run as run_li, simulate_live_phase
from Prop_firm.Trader_launch.calculator_logic.Final_result import run as run_fi
from Common.EOD.trailing_drawdown import aggregate_daily_pnl

ensure_directories()
trades = load_trades(TRADES_CSV)

# ── Part 1: Block size sweep ──
print("=== Block Size Sweep ===")
print(f"{'Block':>6} {'EV':>10} {'ChPass%':>8} {'FuPass%':>8} {'ProbLive%':>10} {'LiveProfit':>12}")
print("-" * 60)

for bs in [1, 2, 3, 5, 7, 10, 15, 20]:
    run_cr(trades=trades)
    ch = run_ch(trades=trades, mc_seed=42, mc_block_size=bs, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
    run_fr(trades=trades)
    fu = run_fu(trades=trades, challenge_result=ch, mc_seed=42, mc_block_size=bs, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
    li = run_li(trades=trades, funded_result=fu, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
    run_fi(challenge_result=ch, funded_result=fu, live_result=li, reports_dir=RESULTS_DIR)

    with open(RESULTS_DIR / "Final_result_metadata.yaml") as f:
        meta = yaml.safe_load(f)

    ev = meta.get("final", {}).get("expected_value_per_pipeline_usd") or 0
    ch_pr = meta.get("challenge", {}).get("mc_pass_rate_pct", 0)
    fu_pr = meta.get("funded", {}).get("mc_pass_rate_pct", 0)
    prob = meta.get("final", {}).get("prob_reaching_live_pct", 0)
    lp = meta.get("live", {}).get("trader_profit_usd", 0)
    print(f"{bs:>6} ${ev:>9.2f} {ch_pr:>7.2f}% {fu_pr:>7.2f}% {prob:>9.2f}% ${lp:>11.2f}")

# ── Part 2: Live phase buffer sweep (using best block size) ──
print("\n=== Live Phase Buffer Sweep (block_size=10) ===")

# Get funded result with block_size=10
run_cr(trades=trades)
ch = run_ch(trades=trades, mc_seed=42, mc_block_size=10, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)
run_fr(trades=trades)
fu = run_fu(trades=trades, challenge_result=ch, mc_seed=42, mc_block_size=10, reports_dir=RESULTS_DIR, figures_dir=FIGURES_DIR)

# Get live phase daily PnL from funded result
funded_summary = fu.get("summary", {})
funded_passed_day = funded_summary.get("funded_day_target")
funded_daily = fu.get("daily_pnl")

if funded_passed_day and funded_daily is not None:
    funded_pass_date = funded_daily.index[funded_passed_day - 1]
    live_start = (pd.Timestamp(funded_pass_date) + pd.Timedelta(days=1)).tz_localize("UTC")
    live_trades = trades[trades.index.normalize() >= live_start].copy()
    live_daily_pnl = aggregate_daily_pnl(live_trades)
else:
    # Fallback: use last funded day
    fallback_day = min(fu.get("deterministic_run").days_traded, len(funded_daily))
    funded_end = funded_daily.index[fallback_day - 1]
    live_start = (pd.Timestamp(funded_end) + pd.Timedelta(days=1)).tz_localize("UTC")
    live_trades = trades[trades.index.normalize() >= live_start].copy()
    live_daily_pnl = aggregate_daily_pnl(live_trades)

print(f"Live phase daily PnL: {len(live_daily_pnl)} days, total ${live_daily_pnl.sum():.2f}")

# Simulate different buffer levels
START = 101_000.0
TERM = 100_000.0
SPLIT = 0.55

print(f"{'Buffer':>8} {'Threshold':>10} {'Withdrawn':>10} {'Profit':>10} {'Days':>6} {'Status':>10} {'#W':>4}")
print("-" * 60)

for buffer_k in [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 104.0, 105.0]:
    buffer = buffer_k * 1000
    threshold = buffer + 200
    balance = START
    total_w = 0.0
    total_p = 0.0
    n_w = 0
    status = "active"
    days = 0

    for i in range(len(live_daily_pnl)):
        balance += float(live_daily_pnl.iloc[i])
        if balance <= TERM:
            status = "terminated"
            days = i + 1
            break
        if balance > threshold:
            withdraw = balance - buffer
            total_w += withdraw
            total_p += round(withdraw * SPLIT, 2)
            n_w += 1
            balance = buffer
        days = i + 1

    if status == "active":
        days = len(live_daily_pnl)

    print(f"${buffer_k:>7.1f}k ${threshold:>9.0f} ${total_w:>9.2f} ${total_p:>9.2f} {days:>6} {status:>10} {n_w:>4}")
