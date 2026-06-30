"""5-year test with floor-aware sizing comparison."""
import sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/root/slop/Prop_firm_EV_calculator')

import numpy as np
import pandas as pd
from Common.EOD.trailing_drawdown import simulate_pnl_sequence
from Common.Monte_carlo.simulator import run_simulations, _auto_block_size

nq = pd.read_csv('/root/slop/Prop_firm_EV_calculator/data/raw/nq_daily_5yr.csv', index_col=0, parse_dates=True)
returns = nq['daily_return'].dropna().values

trader_std = 498.0
nq_std = returns.std()
scale = trader_std / (100_000 * nq_std)
synthetic_pnl = returns * 100_000 * scale

window_size = 60
eval_fee = 45.0
target = 2000.0
dd = 1000.0

# Compare floor-aware vs standard
print(f"{'Start':>10} {'Standard%':>10} {'FloorAware%':>12} {'Std EV':>10} {'FA EV':>10} {'Delta':>8}")
print("-" * 65)

std_rates = []
fa_rates = []
std_evs = []
fa_evs = []

for start in range(0, len(synthetic_pnl) - window_size + 1, 20):
    window = synthetic_pnl[start:start + window_size]
    date_label = f"{nq.index[start].strftime('%Y-%m')}"
    
    # Standard MC
    mc_std = run_simulations(
        pd.Series(window), 100_000, dd, target,
        n_simulations=2000, min_trading_days=3, seed=42, block_size=0,
    )
    
    # Floor-aware MC (pass through to simulate_pnl_sequence)
    # We need to call simulate_pnl_sequence directly with floor_aware=True
    # But run_simulations doesn't pass floor_aware through yet
    # Let's do it manually
    rng = np.random.default_rng(42)
    bs = _auto_block_size(pd.Series(window))
    n = len(window)
    fa_pass = 0
    for _ in range(2000):
        blocks = []
        pos = 0
        while pos < n:
            s = rng.integers(0, n)
            blocks.append(window[(np.arange(s, s + bs) % n)])
            pos += bs
        shuffled = np.concatenate(blocks)[:n]
        result = simulate_pnl_sequence(shuffled, 100_000, dd, target, min_trading_days=3, floor_aware=True)
        if result.passed:
            fa_pass += 1
    fa_rate = 100.0 * fa_pass / 2000
    
    std_rate = mc_std.pass_rate
    std_ev = std_rate/100 * target - (1 - std_rate/100) * eval_fee
    fa_ev = fa_rate/100 * target - (1 - fa_rate/100) * eval_fee
    
    std_rates.append(std_rate)
    fa_rates.append(fa_rate)
    std_evs.append(std_ev)
    fa_evs.append(fa_ev)
    
    print(f"{date_label:>10} {std_rate:>9.1f}% {fa_rate:>11.1f}% ${std_ev:>9.0f} ${fa_ev:>9.0f} {fa_rate - std_rate:>+7.1f}%")

# Summary
std_arr = np.array(std_rates)
fa_arr = np.array(fa_rates)
std_ev_arr = np.array(std_evs)
fa_ev_arr = np.array(fa_evs)

print(f"\n--- 5-Year Comparison ---")
print(f"  Standard:  pass={std_arr.mean():.1f}% ± {std_arr.std():.1f}%, EV=${std_ev_arr.mean():.0f} ± ${std_ev_arr.std():.0f}")
print(f"  Floor-Aware: pass={fa_arr.mean():.1f}% ± {fa_arr.std():.1f}%, EV=${fa_ev_arr.mean():.0f} ± ${fa_ev_arr.std():.0f}")
print(f"  Improvement: {(fa_arr.mean() - std_arr.mean()):+.1f}pp pass rate, ${fa_ev_arr.mean() - std_ev_arr.mean():+.0f} EV")
print(f"  Profitable windows: standard={np.sum(std_ev_arr > 0)}/{len(std_ev_arr)}, floor-aware={np.sum(fa_ev_arr > 0)}/{len(fa_ev_arr)}")
