#!/usr/bin/env python3
"""Multithreaded parameter sweep for VWAP strategy optimization."""
import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from strategy.vwap_strategy import (
    load_nq_data, generate_trades, trades_to_dataframe,
    StrategyParams, MNQ_TICK_VALUE, TICK_SIZE, COMMISSION_PER_SIDE,
)
from Common.EOD.trailing_drawdown import aggregate_daily_pnl
from strategy.sequential_mc import run_sequential_simulation

# Global data — set by initializer
_DF = None

def _init_worker(df_path, cutoff_date):
    global _DF
    _DF = load_nq_data(df_path)
    _DF = _DF[_DF.index >= cutoff_date].copy()

def evaluate_params(combo):
    global _DF
    bar, stop, session, delay, contracts = combo
    
    params = StrategyParams(
        entry_delay_bars=delay,
        stop_atr_fraction=stop,
        session_filter=session,
        contracts=contracts,
        commission_per_side=COMMISSION_PER_SIDE,
    )
    
    trades = generate_trades(_DF, params, bar_minutes=bar)
    if not trades:
        return {'bar': bar, 'stop': stop, 'session': session, 'delay': delay,
                'contracts': contracts, 'n_trades': 0, 'ev': -45,
                'chal_pct': 0, 'fund_pct': 0, 'pipeline_pct': 0,
                'avg_trader': 0, 'daily_std': 0}
    
    tdf = trades_to_dataframe(trades)
    tdf_idx = tdf.set_index('entry_time')
    daily = aggregate_daily_pnl(tdf_idx)
    
    if len(daily) < 10:
        return {'bar': bar, 'stop': stop, 'session': session, 'delay': delay,
                'contracts': contracts, 'n_trades': len(tdf), 'ev': -45,
                'chal_pct': 0, 'fund_pct': 0, 'pipeline_pct': 0,
                'avg_trader': 0, 'daily_std': 0}
    
    result = run_sequential_simulation(daily_pnl=daily.values, n_simulations=3000, seed=42)
    
    return {
        'bar': bar, 'stop': stop, 'session': session, 'delay': delay,
        'contracts': contracts,
        'n_trades': len(tdf),
        'daily_std': float(daily.std()),
        'chal_pct': result.challenge_pass_rate,
        'fund_pct': result.funded_pass_rate,
        'pipeline_pct': result.pipeline_pass_rate,
        'avg_trader': result.avg_trader_profit,
        'ev': result.ev_per_pipeline,
    }


def main():
    nq_path = PROJECT_ROOT / "data" / "raw" / "nq-1m.csv"
    cutoff = "2020-01-01"
    
    combos = []
    for bar in [5, 15]:
        for stop in [0.25, 0.50, 1.0, 2.0]:
            for session in ["full", "morning"]:
                for delay in [1, 2]:
                    for contracts in [1, 2, 3, 5]:
                        combos.append((bar, stop, session, delay, contracts))
    
    print(f"Running {len(combos)} combinations with {os.cpu_count()} cores...")
    t0 = time.time()
    
    results = []
    with ProcessPoolExecutor(
        max_workers=min(os.cpu_count(), 8),
        initializer=_init_worker,
        initargs=(str(nq_path), cutoff),
    ) as pool:
        futures = {pool.submit(evaluate_params, c): c for c in combos}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  ERROR at {done}: {e}")
            if done % 20 == 0 or done == len(combos):
                elapsed = time.time() - t0
                best_so_far = max(results, key=lambda r: r['ev']) if results else None
                best_str = f", best EV=${best_so_far['ev']:.0f}" if best_so_far else ""
                print(f"  {done}/{len(combos)} done ({elapsed:.0f}s{best_str})")
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    
    res = pd.DataFrame(results).sort_values('ev', ascending=False)
    
    print(f"\n{'bar':>3} {'stop':>5} {'sess':>7} {'d':>2} {'c':>2} {'n':>5} {'std':>6} {'chal%':>6} {'fund%':>6} {'pipe%':>6} {'avg$':>7} {'EV':>8}")
    print("-" * 80)
    for _, r in res.head(20).iterrows():
        print(f"{r.bar:>3.0f} {r.stop:>5.2f} {r.session:>7} {r.delay:>2.0f} {r.contracts:>2.0f} "
              f"{r.n_trades:>5.0f} ${r.daily_std:>5.0f} {r.chal_pct:>5.1f}% {r.fund_pct:>5.1f}% "
              f"{r.pipeline_pct:>5.1f}% ${r.avg_trader:>6.0f} ${r.ev:>7.0f}")
    
    b = res.iloc[0]
    print(f"\nBEST: bar={b.bar:.0f}m, stop={b.stop}, session={b.session}, delay={b.delay:.0f}, contracts={b.contracts:.0f}")
    print(f"  EV: ${b.ev:.0f}/pipeline")
    print(f"  Challenge: {b.chal_pct:.1f}%, Funded(of chal): {b.fund_pct:.1f}%, Pipeline: {b.pipeline_pct:.1f}%")
    print(f"  Avg trader profit: ${b.avg_trader:.0f}")
    
    # Save results
    res.to_csv(PROJECT_ROOT / "sweep_results.csv", index=False)
    print(f"\nResults saved to sweep_results.csv")


if __name__ == "__main__":
    main()
