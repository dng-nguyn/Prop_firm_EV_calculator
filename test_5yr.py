"""5-year out-of-sample test using NQ futures daily returns.

Generates synthetic daily PnL from real NQ returns, then tests
pass rate stability across 60-day windows over 5 years.
"""
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/root/slop/Prop_firm_EV_calculator')

import numpy as np
import pandas as pd
from Common.EOD.trailing_drawdown import simulate_pnl_sequence
from Common.Monte_carlo.simulator import run_simulations, _auto_block_size

# Load 5-year NQ data
nq = pd.read_csv('/root/slop/Prop_firm_EV_calculator/data/raw/nq_daily_5yr.csv', index_col=0, parse_dates=True)
returns = nq['daily_return'].dropna().values

print("=" * 70)
print("5-YEAR OUT-OF-SAMPLE TEST — NQ FUTURES")
print("=" * 70)
print(f"Data: {len(returns)} trading days ({nq.index.min().date()} to {nq.index.max().date()})")
print(f"Mean daily return: {returns.mean()*100:.4f}%")
print(f"Std daily return: {returns.std()*100:.4f}%")

# Generate synthetic daily PnL
# Assume: $100k account, ~2% risk per trade, NQ micro ($2/point)
# Daily PnL = account * return * risk_scaling
# The trader's data shows mean daily PnL of $10.78 with std $498
# On $100k, that's ~0.01% mean with ~0.5% std
# NQ daily returns have ~1.43% std, so we need to scale down
# Scale: daily_pnl = return * account * (trader_std / nq_std)
trader_std = 498.0  # from actual data
nq_std = returns.std()
scale = trader_std / (100_000 * nq_std)  # fraction of account per NQ return

print(f"\nSynthetic PnL scaling: {scale:.6f} (maps NQ returns to trader's PnL distribution)")
print(f"Expected daily PnL std: ${100_000 * nq_std * scale:.0f}")

# Generate synthetic daily PnL for the full 5-year period
synthetic_daily_pnl = returns * 100_000 * scale

# ── Walk-forward test across 60-day windows ──
print(f"\n--- Walk-Forward: 60-Day Challenge Windows ---")
window_size = 60
challenge_target = 2_000.0
max_dd = 1_000.0

pass_rates = []
ev_values = []
window_labels = []

for start in range(0, len(synthetic_daily_pnl) - window_size + 1, 20):  # every 20 days
    window = synthetic_daily_pnl[start:start + window_size]
    
    # Auto block size for this window
    bs = _auto_block_size(pd.Series(window))
    
    # Run MC
    mc = run_simulations(
        pd.Series(window), 100_000, max_dd, challenge_target,
        n_simulations=3000, min_trading_days=3, seed=42, block_size=bs,
    )
    
    ev = mc.pass_rate/100 * challenge_target - (1 - mc.pass_rate/100) * 45
    
    date_label = f"{nq.index[start].strftime('%Y-%m')}"
    pass_rates.append(mc.pass_rate)
    ev_values.append(ev)
    window_labels.append(date_label)

# Print results
print(f"{'Start':>10} {'Pass%':>8} {'EV':>10} {'Block':>6}")
print("-" * 36)
for label, pr, ev in zip(window_labels, pass_rates, ev_values):
    print(f"{label:>10} {pr:>7.1f}% ${ev:>9.0f}")

# ── Summary statistics ──
pr_arr = np.array(pass_rates)
ev_arr = np.array(ev_values)

print(f"\n--- 5-Year Summary ---")
print(f"  Windows tested: {len(pass_rates)}")
print(f"  Pass rate: {pr_arr.mean():.1f}% ± {pr_arr.std():.1f}% (range: {pr_arr.min():.1f}%–{pr_arr.max():.1f}%)")
print(f"  EV: ${ev_arr.mean():.0f} ± ${ev_arr.std():.0f} (range: ${ev_arr.min():.0f}–${ev_arr.max():.0f})")
print(f"  Profitable windows: {(ev_arr > 0).sum()}/{len(ev_arr)} ({(ev_arr > 0).mean()*100:.0f}%)")

# ── Overfitting check ──
print(f"\n--- Overfitting Diagnostic ---")
# Compare first half vs second half
half = len(pass_rates) // 2
first_half = pr_arr[:half]
second_half = pr_arr[half:]
print(f"  First half (2021-2023): {first_half.mean():.1f}% ± {first_half.std():.1f}%")
print(f"  Second half (2023-2026): {second_half.mean():.1f}% ± {second_half.std():.1f}%")
print(f"  Stability ratio: {second_half.mean() / first_half.mean():.2f} (1.0 = perfectly stable)")

# ── Year-by-year breakdown ──
print(f"\n--- Year-by-Year ---")
for year in range(2021, 2027):
    year_mask = [nq.index[i].year == year for i in range(0, len(pass_rates) * 20, 20)]
    if sum(year_mask) == 0:
        continue
    year_rates = pr_arr[year_mask]
    year_evs = ev_arr[year_mask]
    print(f"  {year}: pass={year_rates.mean():.1f}% ± {year_rates.std():.1f}%, EV=${year_evs.mean():.0f} ± ${year_evs.std():.0f}")
