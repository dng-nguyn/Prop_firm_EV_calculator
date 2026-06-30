"""Universal EV Calculator: expanding-window out-of-sample validation.

Tests whether the trader's edge persists across time, not just on one dataset.
Uses expanding windows (train on days 1..t, test on days t+1..t+k) to simulate
realistic out-of-sample performance.
"""
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/root/slop/Prop_firm_EV_calculator')

import numpy as np
import pandas as pd
from Prop_firm.Trader_launch.Trader_launch_calculator import load_trades, TRADES_CSV
from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import apply_daily_lock_rule, _load_rules
from Common.EOD.trailing_drawdown import aggregate_daily_pnl, simulate_pnl_sequence
from Common.Monte_carlo.simulator import run_simulations, _auto_block_size

trades = load_trades(TRADES_CSV)
rules = _load_rules()
cfg = rules["phases"]["challenge"]
profit_target = float(cfg["profit_target"])
consistency_pct = float(cfg["consistency_rule"])
eval_fee = float(rules.get("general", {}).get("evaluation_fee", 45))

# Apply lock rule
trades_filtered, stats = apply_daily_lock_rule(trades, profit_target, consistency_pct)
daily_pnl = aggregate_daily_pnl(trades_filtered)
pnl_values = daily_pnl.values
n = len(pnl_values)

print("=" * 70)
print("UNIVERSAL EV CALCULATOR — OUT-OF-SAMPLE VALIDATION")
print("=" * 70)

# ── 1. Expanding-window walk-forward ──
# Train on days 1..t, test pass rate on days t+1..t+window
print(f"\n--- Expanding-Window Walk-Forward ---")
print(f"{'Train End':>10} {'Test Window':>12} {'Train Pass%':>12} {'Test Pass%':>12} {'Train EV':>10} {'Test EV':>10}")
print("-" * 70)

train_window = 30  # minimum training data
test_window = 15   # test on next 15 days
min_train = 20

oos_results = []
for train_end in range(train_window, n - test_window + 1, 5):
    train_data = pnl_values[:train_end]
    test_data = pnl_values[train_end:train_end + test_window]
    
    if len(test_data) < 10:
        continue
    
    # Auto block size for each
    train_bs = _auto_block_size(pd.Series(train_data))
    test_bs = _auto_block_size(pd.Series(test_data))
    
    # Train pass rate (in-sample)
    mc_train = run_simulations(
        pd.Series(train_data), 100_000, 1_000, 2_000,
        n_simulations=3000, min_trading_days=3, seed=42, block_size=train_bs,
    )
    
    # Test pass rate (out-of-sample)
    mc_test = run_simulations(
        pd.Series(test_data), 100_000, 1_000, 2_000,
        n_simulations=3000, min_trading_days=3, seed=42, block_size=test_bs,
    )
    
    train_ev = mc_train.pass_rate/100 * profit_target - (1-mc_train.pass_rate/100) * eval_fee
    test_ev = mc_test.pass_rate/100 * profit_target - (1-mc_test.pass_rate/100) * eval_fee
    
    oos_results.append({
        'train_end': train_end,
        'train_pass': mc_train.pass_rate,
        'test_pass': mc_test.pass_rate,
        'train_ev': train_ev,
        'test_ev': test_ev,
    })
    
    print(f"Day {train_end:>4}    {train_end+1:>3}-{train_end+test_window:<3}    {mc_train.pass_rate:>10.1f}%  {mc_test.pass_rate:>10.1f}%  ${train_ev:>8.0f}  ${test_ev:>8.0f}")

# ── 2. Summary statistics ──
train_passes = [r['train_pass'] for r in oos_results]
test_passes = [r['test_pass'] for r in oos_results]
train_evs = [r['train_ev'] for r in oos_results]
test_evs = [r['test_ev'] for r in oos_results]

print(f"\n--- Out-of-Sample Summary ---")
print(f"  Train pass rate: {np.mean(train_passes):.1f}% ± {np.std(train_passes):.1f}% (range: {np.min(train_passes):.1f}%–{np.max(train_passes):.1f}%)")
print(f"  Test pass rate:  {np.mean(test_passes):.1f}% ± {np.std(test_passes):.1f}% (range: {np.min(test_passes):.1f}%–{np.max(test_passes):.1f}%)")
print(f"  Train EV: ${np.mean(train_evs):.0f} ± ${np.std(train_evs):.0f}")
print(f"  Test EV:  ${np.mean(test_evs):.0f} ± ${np.std(test_evs):.0f}")

# ── 3. Overfitting ratio ──
# If test_pass << train_pass, the model is overfit
avg_train = np.mean(train_passes)
avg_test = np.mean(test_passes)
overfit_ratio = avg_test / avg_train if avg_train > 0 else 0

print(f"\n--- Overfitting Diagnostic ---")
print(f"  Train/Test ratio: {overfit_ratio:.2f} (1.0 = no overfit, <0.7 = severe overfit)")
if overfit_ratio > 0.85:
    print(f"  ✓ Edge appears robust out-of-sample")
elif overfit_ratio > 0.7:
    print(f"  ⚠ Moderate overfit — edge may not persist")
else:
    print(f"  ✗ Severe overfit — edge is likely spurious")

# ── 4. Cumulative OOS performance ──
print(f"\n--- Cumulative Out-of-Sample EV ---")
cumulative_test_ev = 0
for r in oos_results:
    cumulative_test_ev += r['test_ev']
print(f"  Sum of OOS EVs: ${cumulative_test_ev:.0f}")
print(f"  Average OOS EV per window: ${cumulative_test_ev/len(oos_results):.0f}")

# ── 5. The honest answer ──
print(f"\n--- The Honest Answer ---")
print(f"  In-sample (full dataset): pass rate = {avg_train:.1f}%")
print(f"  Out-of-sample (walk-forward): pass rate = {avg_test:.1f}%")
print(f"  The edge {'persists' if overfit_ratio > 0.85 else 'degrades'} out-of-sample.")
print(f"  A trader using this calculator should expect ~{avg_test:.0f}% pass rate,")
print(f"  not the in-sample {avg_train:.0f}%.")
