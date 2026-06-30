"""Out-of-sample validation: test if block=28 is optimal across different time windows."""
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/root/slop/Prop_firm_EV_calculator')

import numpy as np
import pandas as pd
from Common.EOD.trailing_drawdown import aggregate_daily_pnl, simulate_pnl_sequence
from Prop_firm.Trader_launch.calculator_logic.Challenge_phase import apply_daily_lock_rule, _load_rules
from Prop_firm.Trader_launch.Trader_launch_calculator import load_trades, TRADES_CSV

trades = load_trades(TRADES_CSV)
rules = _load_rules()
cfg = rules["phases"]["challenge"]
profit_target = float(cfg["profit_target"])
consistency_pct = float(cfg["consistency_rule"])

trades_filtered, _ = apply_daily_lock_rule(trades, profit_target, consistency_pct)
daily_pnl = aggregate_daily_pnl(trades_filtered)
pnl_values = daily_pnl.values
n = len(pnl_values)

def mc_block(pnl_values, n_sims, block_size, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl_values)
    pass_count = 0
    for _ in range(n_sims):
        blocks = []
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            block = pnl_values[(np.arange(start, start + block_size) % n)]
            blocks.append(block)
            pos += block_size
        shuffled = np.concatenate(blocks)[:n]
        result = simulate_pnl_sequence(shuffled, 100_000, 1_000, 2_000, min_trading_days=3)
        if result.passed:
            pass_count += 1
    return 100.0 * pass_count / n_sims

# ── Walk-forward out-of-sample test ──
# Split into train/test windows, find optimal block on train, test on test
print("=== Walk-Forward Out-of-Sample Validation ===")
print(f"{'Window':<15} {'Train Optimal':>14} {'Train Pass%':>12} {'Test Pass%':>12} {'Test Best BS':>13}")
print("-" * 70)

block_sizes = [5, 7, 10, 15, 20, 25, 28, 30, 35]
window_size = 30

for start in range(0, n - window_size + 1, 5):
    window = pnl_values[start:start + window_size]
    
    # Split 60/40 train/test within the window
    split = int(window_size * 0.6)
    train = window[:split]
    test = window[split:]
    
    if len(test) < 10:
        continue
    
    # Find optimal block on train
    train_results = {}
    for bs in block_sizes:
        if bs > len(train):
            continue
        pr = mc_block(train, 3000, bs, seed=42)
        train_results[bs] = pr
    
    best_train_bs = max(train_results, key=train_results.get)
    best_train_pr = train_results[best_train_bs]
    
    # Test all blocks on test set
    test_results = {}
    for bs in block_sizes:
        if bs > len(test):
            continue
        pr = mc_block(test, 3000, bs, seed=42)
        test_results[bs] = pr
    
    best_test_bs = max(test_results, key=test_results.get)
    test_pr_at_train_bs = test_results.get(best_train_bs, 0)
    
    window_label = f"Days {start}-{start+window_size}"
    print(f"{window_label:<15} bs={best_train_bs:>2} ({best_train_pr:.1f}%)  {test_pr_at_train_bs:>10.1f}%  {test_results[best_test_bs]:>10.1f}% (bs={best_test_bs})")

# ── Full dataset: block size sensitivity ──
print(f"\n=== Full Dataset: Block Size Sensitivity ===")
print(f"{'Block':>6} {'Pass%':>8} {'Delta vs 28':>12}")
print("-" * 28)
pr_28 = mc_block(pnl_values, 5000, 28, seed=42)
for bs in [5, 10, 15, 20, 25, 28, 30, 35, 40]:
    pr = mc_block(pnl_values, 5000, bs, seed=42)
    print(f"{bs:>6} {pr:>7.2f}% {pr - pr_28:>+11.2f}%")

# ── Cross-validation: which block size wins most often? ===
print(f"\n=== Cross-Validation: Block Size Win Count ===")
print("(Which block size gives highest pass rate across different random seeds?)")

win_count = {bs: 0 for bs in block_sizes}
for seed in range(1, 51):  # 50 different seeds
    best_bs = None
    best_pr = 0
    for bs in block_sizes:
        pr = mc_block(pnl_values, 2000, bs, seed=seed)
        if pr > best_pr:
            best_pr = pr
            best_bs = bs
    win_count[best_bs] += 1

print(f"{'Block':>6} {'Wins':>6} {'Win%':>6}")
print("-" * 20)
for bs in sorted(win_count.keys()):
    print(f"{bs:>6} {win_count[bs]:>6} {win_count[bs]/50*100:>5.1f}%")
