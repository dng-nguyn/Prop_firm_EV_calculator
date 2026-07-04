"""
Benchmark: VWAP Strategy on NQ Futures for Prop Firm EV
========================================================
Optimized: vectorized MC, multiprocessing sweep, proper phase separation.
"""

from __future__ import annotations

import sys
import gc
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Prop firm rules (Trader Launch)
# ═══════════════════════════════════════════════════════════════════════
FEE = 45.0
START_BAL = 100_000.0
CHAL_DD = 1_000.0
CHAL_TP = 2_000.0
FUND_DD = 1_000.0
FUND_TP = 1_000.0
CONSISTENCY_PCT = 0.40
N_MC = 5_000
LIVE_START = 101_000.0
LIVE_TERM = 100_000.0
LIVE_THRESH = 101_200.0
LIVE_BUF = 101_000.0
PROFIT_SPLIT = 0.55


# ═══════════════════════════════════════════════════════════════════════
# Vectorized MC — all simulations in one numpy pass
# ═══════════════════════════════════════════════════════════════════════

def vectorized_mc(
    daily_pnl: np.ndarray,
    start_balance: float,
    max_drawdown: float,
    profit_target: float,
    n_simulations: int = 5000,
    seed: int = 42,
    min_trading_days: int = 3,
    return_pass_days: bool = False,
) -> dict:
    """
    Run all MC simulations vectorized — no Python for-loop.
    Returns pass_rate, avg_pass_days, avg_fail_days.
    """
    n_days = len(daily_pnl)
    if n_days == 0:
        return {"pass_rate": 0, "avg_pass_days": 0, "avg_fail_days": 0, "n": n_simulations}

    rng = np.random.default_rng(seed)
    
    # Pre-shuffle all paths at once: shape (n_simulations, n_days)
    # Use argsort trick for fast shuffling
    randoms = rng.random((n_simulations, n_days), dtype=np.float32)
    shuffle_idx = np.argsort(randoms, axis=1)
    all_pnl = daily_pnl[shuffle_idx]  # (n_simulations, n_days)
    
    # Vectorized EOD simulation
    cum_pnl = np.zeros(n_simulations, dtype=np.float64)
    peak = np.full(n_simulations, start_balance, dtype=np.float64)
    floor = np.full(n_simulations, start_balance - max_drawdown, dtype=np.float64)
    floor_locked = np.zeros(n_simulations, dtype=bool)
    lock_threshold = start_balance + max_drawdown
    
    passed = np.zeros(n_simulations, dtype=bool)
    breached = np.zeros(n_simulations, dtype=bool)
    pass_day = np.zeros(n_simulations, dtype=np.int32)
    fail_day = np.zeros(n_simulations, dtype=np.int32)
    active = np.ones(n_simulations, dtype=bool)
    
    for day in range(n_days):
        if not active.any():
            break
        
        day_pnl = all_pnl[:, day]
        cum_pnl += day_pnl
        equity = start_balance + cum_pnl
        
        # Update peak
        new_peak = equity > peak
        peak = np.where(new_peak, equity, peak)
        
        # Update floor
        should_lock = (~floor_locked) & (peak >= lock_threshold)
        floor = np.where(should_lock, start_balance, floor)
        floor_locked = floor_locked | should_lock
        
        trail_update = (~floor_locked) & ((peak - max_drawdown) > floor)
        floor = np.where(trail_update, peak - max_drawdown, floor)
        
        # Check breach: equity < floor
        breach = active & (~passed) & (equity < floor)
        breached |= breach
        fail_day = np.where(breach & (fail_day == 0), day + 1, fail_day)
        active &= ~breach
        
        # Check profit target (only after min_trading_days)
        hit_target = active & (~breached) & (cum_pnl >= profit_target) & (day + 1 >= min_trading_days)
        passed |= hit_target
        pass_day = np.where(hit_target & (pass_day == 0), day + 1, pass_day)
        active &= ~hit_target
    
    n_pass = passed.sum()
    n_fail = breached.sum()
    
    pass_rate = 100.0 * n_pass / n_simulations
    
    pass_days = pass_day[pass_day > 0]
    fail_days = fail_day[fail_day > 0]
    
    result = {
        "pass_rate": pass_rate,
        "avg_pass_days": float(pass_days.mean()) if len(pass_days) > 0 else 0,
        "avg_fail_days": float(fail_days.mean()) if len(fail_days) > 0 else 0,
        "n": n_simulations,
    }
    if return_pass_days:
        result["pass_day_arr"] = pass_day  # per-sim pass day (0 = didn't pass)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Trade generation (VWAP strategy)
# ═══════════════════════════════════════════════════════════════════════

def load_nq_data(csv_path: str, year_start: int = 2020) -> pd.DataFrame:
    """Load NQ 1-min data, subset to year_start+."""
    df = pd.read_csv(
        csv_path, sep=";", header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"open": "float32", "high": "float32", "low": "float32",
               "close": "float32", "volume": "float32"},
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    df = df[df.index >= f"{year_start}-01-01"].copy()
    df.sort_index(inplace=True)
    return df


def generate_daily_pnl(
    df: pd.DataFrame,
    bar_minutes: int = 5,
    stop_atr_fraction: float = 1.0,
    session: str = "full",
    contracts: int = 2,
    commission: float = 1.50,
    tick_size: float = 0.25,
    tick_value: float = 5.0,
    entry_delay: int = 0,
) -> pd.Series:
    """
    Generate daily PnL from VWAP strategy directly (no Trade objects).
    Returns pd.Series indexed by date with daily net PnL.
    """
    from datetime import time as dtime
    
    # Resample to bars
    if bar_minutes > 1:
        bars = df.resample(f"{bar_minutes}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    else:
        bars = df
    
    n = len(bars)
    if n < 100:
        return pd.Series(dtype=float)
    
    # Compute intraday VWAP
    typical = ((bars["high"].astype("float64") + bars["low"].astype("float64") + bars["close"].astype("float64")) / 3.0).values
    vol = bars["volume"].astype("float64").values
    dates = bars.index.date
    date_changes = np.concatenate([[True], dates[1:] != dates[:-1]])
    group_id = np.cumsum(date_changes) - 1
    
    cum_tp_vol = np.zeros(n)
    cum_vol = np.zeros(n)
    for g in range(group_id[-1] + 1):
        mask = group_id == g
        cum_tp_vol[mask] = np.cumsum(typical[mask] * vol[mask])
        cum_vol[mask] = np.cumsum(vol[mask])
    
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    
    # Compute ATR (daily)
    daily_high = np.zeros(group_id[-1] + 1)
    daily_low = np.zeros(group_id[-1] + 1)
    daily_close = np.zeros(group_id[-1] + 1)
    for g in range(group_id[-1] + 1):
        mask = group_id == g
        daily_high[g] = bars["high"].values[mask].max()
        daily_low[g] = bars["low"].values[mask].min()
        daily_close[g] = bars["close"].values[mask][-1]
    
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - np.roll(daily_close, 1)),
                               np.abs(daily_low - np.roll(daily_close, 1))))
    tr[0] = daily_high[0] - daily_low[0]
    
    # EWM ATR
    atr_daily = np.zeros(len(tr))
    atr_daily[0] = tr[0]
    alpha = 2.0 / (14 + 1)
    for i in range(1, len(tr)):
        atr_daily[i] = alpha * tr[i] + (1 - alpha) * atr_daily[i - 1]
    
    # Map ATR to intraday
    atr = atr_daily[group_id]
    
    # Session filter
    times = bars.index.time
    if session == "morning":
        session_mask = np.array([(dtime(9, 30) <= t <= dtime(12, 0)) for t in times])
    else:
        session_mask = np.array([(dtime(9, 30) <= t <= dtime(15, 45)) for t in times])
    
    eod_mask = np.array([t >= dtime(16, 0) for t in times])
    
    # Generate trades
    close = bars["close"].values.astype("float64")
    high = bars["high"].values.astype("float64")
    low = bars["low"].values.astype("float64")
    
    daily_pnl_dict = {}
    
    in_position = False
    side = 0  # 1=long, -1=short
    entry_px = 0.0
    stop_px = 0.0
    entry_day = None
    pending_signal = 0
    signal_count = 0
    
    for i in range(n):
        cur_date = dates[i]
        cur_vwap = vwap[i]
        cur_atr = atr[i]
        
        if np.isnan(cur_vwap) or cur_vwap <= 0:
            continue
        
        # EOD exit
        if in_position and eod_mask[i]:
            pnl_ticks = (close[i] - entry_px) / tick_size * side
            pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
            daily_pnl_dict.setdefault(entry_day, 0.0)
            daily_pnl_dict[cur_date] = daily_pnl_dict.get(cur_date, 0.0) + pnl_dollars
            in_position = False
            continue
        
        # Check stop
        if in_position:
            hit_stop = False
            if side == 1 and low[i] <= stop_px:
                exit_px = stop_px
                hit_stop = True
            elif side == -1 and high[i] >= stop_px:
                exit_px = stop_px
                hit_stop = True
            
            if hit_stop:
                pnl_ticks = (exit_px - entry_px) / tick_size * side
                pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
                daily_pnl_dict[cur_date] = daily_pnl_dict.get(cur_date, 0.0) + pnl_dollars
                in_position = False
            continue
        
        # Signal
        if not session_mask[i]:
            continue
        
        signal = 1 if close[i] > cur_vwap else (-1 if close[i] < cur_vwap else 0)
        if signal == 0:
            continue
        
        # Entry delay
        if signal != pending_signal:
            pending_signal = signal
            signal_count = 0
            continue
        signal_count += 1
        if signal_count <= entry_delay:
            continue
        
        # Enter
        side = signal
        entry_px = close[i]
        entry_day = cur_date
        
        stop_dist = cur_atr * stop_atr_fraction if cur_atr > 0 else 10 * tick_size
        stop_px = entry_px - stop_dist * side
        
        in_position = True
        pending_signal = 0
        signal_count = 0
    
    # Close any remaining position at last bar
    if in_position:
        pnl_ticks = (close[-1] - entry_px) / tick_size * side
        pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
        daily_pnl_dict[entry_day] = daily_pnl_dict.get(entry_day, 0.0) + pnl_dollars
    
    if not daily_pnl_dict:
        return pd.Series(dtype=float)
    
    result = pd.Series(daily_pnl_dict).sort_index()
    result.index = pd.to_datetime(result.index)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Full EV pipeline (properly separates challenge & funded)
# ═══════════════════════════════════════════════════════════════════════

def compute_ev(
    daily_pnl: pd.Series,
    n_mc: int = N_MC,
    fee: float = FEE,
) -> dict:
    """
    Full EV pipeline:
    - Challenge: MC on all daily PnL, target $2000, DD $1000
    - Funded: MC on all daily PnL, target $1000, DD $1000
    - Live: deterministic walk with withdrawals
    - Consistency: check 40% rule
    """
    if len(daily_pnl) < 10:
        return {"ev": -fee, "chal_rate": 0, "fund_rate": 0, "live_profit": 0, "n_days": 0}
    
    pnl = daily_pnl.values.astype(np.float64)
    
    # Consistency check: max profitable day / gross profits <= 40%
    profitable_days = pnl[pnl > 0]
    if len(profitable_days) > 0:
        gross_profit = profitable_days.sum()
        max_day = pnl.max()
        consistency_pass = (max_day / gross_profit) <= CONSISTENCY_PCT if gross_profit > 0 else True
    else:
        consistency_pass = True
    
    # Challenge phase MC (return pass days for funded truncation)
    chal = vectorized_mc(pnl, START_BAL, CHAL_DD, CHAL_TP, n_simulations=n_mc,
                         seed=42, min_trading_days=3, return_pass_days=True)
    chal_rate = chal["pass_rate"] / 100.0
    
    if chal_rate < 0.01:
        return {"ev": -fee, "chal_rate": chal_rate, "fund_rate": 0, "live_profit": 0, "n_days": len(pnl)}
    
    # Funded phase MC: only use days AFTER median challenge pass day
    pass_day_arr = chal["pass_day_arr"]
    passing_days = pass_day_arr[pass_day_arr > 0]
    median_pass_day = int(np.median(passing_days)) if len(passing_days) > 0 else len(pnl) // 2
    funded_pnl = pnl[median_pass_day:]  # days after challenge typically passes
    
    if len(funded_pnl) < 5:
        fund_rate = 0.0
    else:
        fund = vectorized_mc(funded_pnl, START_BAL, FUND_DD, FUND_TP,
                             n_simulations=n_mc, seed=43, min_trading_days=1)
        fund_rate = fund["pass_rate"] / 100.0
    
    # Live phase: deterministic
    bal = LIVE_START
    withdrawn = 0.0
    for p in pnl:
        bal += p
        if bal <= LIVE_TERM:
            break
        if bal >= LIVE_THRESH:
            profit = bal - LIVE_BUF
            if profit > 0:
                withdrawn += profit * PROFIT_SPLIT
                bal = LIVE_BUF
    
    # EV
    pipeline_rate = chal_rate * fund_rate
    ev = pipeline_rate * withdrawn - fee
    
    return {
        "ev": ev,
        "chal_rate": chal_rate,
        "fund_rate": fund_rate,
        "live_profit": withdrawn,
        "consistency_pass": consistency_pass,
        "n_days": len(pnl),
        "n_trades": (pnl != 0).sum(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Parameter sweep (single-core, but uses vectorized MC)
# ═══════════════════════════════════════════════════════════════════════

def sweep_params(df: pd.DataFrame, bar_minutes: int) -> list[dict]:
    """Sweep stop fractions and sessions for a given bar size. d=0 only."""
    results = []
    for stop in [0.25, 0.50, 0.75, 1.0, 1.5, 2.0]:
        for sess in ["full", "morning"]:
            t0 = time.time()
            daily = generate_daily_pnl(
                df, bar_minutes=bar_minutes, stop_atr_fraction=stop,
                session=sess, contracts=2, commission=1.50, entry_delay=0,
            )
            ev_result = compute_ev(daily)
            elapsed = time.time() - t0
            results.append({
                "bar": bar_minutes, "stop": stop, "session": sess,
                "n_days": len(daily), "elapsed": elapsed, **ev_result,
            })
            print(f"  {bar_minutes}m s={stop:.2f} {sess[:5]} → EV=${ev_result['ev']:>7.0f} "
                  f"({ev_result['chal_rate']:.0%}×{ev_result['fund_rate']:.0%} "
                  f"live=${ev_result['live_profit']:>6.0f} n={len(daily)}) [{elapsed:.0f}s]")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Robustness tests (d=1, d=2 on winning params)
# ═══════════════════════════════════════════════════════════════════════

def test_robustness(df: pd.DataFrame, bar: int, stop: float, sess: str) -> dict:
    """Test robustness: delay 1, delay 2, perturbation."""
    # Baseline (d=0)
    base_daily = generate_daily_pnl(df, bar, stop, sess, contracts=2, commission=1.50, entry_delay=0)
    base_ev = compute_ev(base_daily)
    
    # Delay 1
    d1_daily = generate_daily_pnl(df, bar, stop, sess, contracts=2, commission=1.50, entry_delay=1)
    d1_ev = compute_ev(d1_daily)
    
    # Delay 2
    d2_daily = generate_daily_pnl(df, bar, stop, sess, contracts=2, commission=1.50, entry_delay=2)
    d2_ev = compute_ev(d2_daily)
    
    # Perturbation: add ±1 tick noise to daily PnL
    rng = np.random.default_rng(42)
    perturbed_pnls = []
    for _ in range(20):
        noise = rng.normal(0, 2.0, len(base_daily))  # ~$2 noise per day
        perturbed = base_daily.copy()
        perturbed += noise
        perturbed_pnls.append(compute_ev(perturbed)["ev"])
    
    avg_perturb_ev = np.mean(perturbed_pnls)
    
    # Degradation
    base = base_ev["ev"]
    d1_deg = (base - d1_ev["ev"]) / abs(base) if base > 0 else 0
    d2_deg = (base - d2_ev["ev"]) / abs(base) if base > 0 else 0
    perturb_deg = (base - avg_perturb_ev) / abs(base) if base > 0 else 0
    
    return {
        "base_ev": base, "base_chal": base_ev["chal_rate"], "base_fund": base_ev["fund_rate"],
        "d1_ev": d1_ev["ev"], "d1_deg": d1_deg,
        "d2_ev": d2_ev["ev"], "d2_deg": d2_deg,
        "perturb_ev": avg_perturb_ev, "perturb_deg": perturb_deg,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    nq_path = str(PROJECT_ROOT / "data" / "raw" / "nq-1m.csv")
    
    print("=" * 70)
    print("VWAP STRATEGY OPTIMIZATION — PROP FIRM EV CALCULATOR")
    print("=" * 70)
    
    t0 = time.time()
    df = load_nq_data(nq_path, year_start=2020)
    print(f"\nLoaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()}) "
          f"[{time.time()-t0:.0f}s, {df.memory_usage(deep=True).sum()/1e6:.0f}MB]")
    
    # ── Phase 1: Sweep with d=0 ────────────────────────────────────────
    print(f"\n--- PHASE 1: d=0 SWEEP ---")
    r5 = sweep_params(df, 5)
    r15 = sweep_params(df, 15)
    d0_df = pd.DataFrame(r5 + r15).sort_values('ev', ascending=False)
    top3 = d0_df.head(3)
    print(f"\n  Top 3 (d=0):")
    for i, (_, r) in enumerate(top3.iterrows()):
        print(f"    {i+1}. {r.bar}m s={r.stop:.2f} {r.session}: EV=${r.ev:.0f}")
    
    # ── Phase 2: Re-test top 3 with d=1 ────────────────────────────────
    print(f"\n--- PHASE 2: d=1 RETEST ---")
    d1_results = []
    for _, r in top3.iterrows():
        daily = generate_daily_pnl(df, int(r.bar), r.stop, r.session,
                                   contracts=2, commission=1.50, entry_delay=1)
        ev = compute_ev(daily, n_mc=N_MC)
        d1_results.append({"bar": int(r.bar), "stop": r.stop, "session": r.session,
                           "d0_ev": r.ev, "d1_ev": ev["ev"], "d1_chal": ev["chal_rate"],
                           "d1_fund": ev["fund_rate"], "d1_live": ev["live_profit"]})
        print(f"    {int(r.bar)}m s={r.stop:.2f} {r.session}: d0=${r.ev:.0f} → d1=${ev['ev']:.0f}")
    
    winner = max(d1_results, key=lambda x: x["d1_ev"])
    bar, stop, sess = winner["bar"], winner["stop"], winner["session"]
    
    print(f"\n{'='*70}")
    print(f"WINNER: {bar}m stop={stop}×ATR session={sess}")
    print(f"  d=0 EV=${winner['d0_ev']:.0f}, d=1 EV=${winner['d1_ev']:.0f}")
    print(f"{'='*70}")
    
    # ── Phase 3: Full robustness ────────────────────────────────────────
    print(f"\n--- PHASE 3: ROBUSTNESS ---")
    rob = test_robustness(df, bar, stop, sess)
    print(f"  d=0: EV=${rob['base_ev']:.0f} ({rob['base_chal']:.0%}×{rob['base_fund']:.0%})")
    print(f"  d=1: EV=${rob['d1_ev']:.0f} (deg: {rob['d1_deg']:.0%})")
    print(f"  d=2: EV=${rob['d2_ev']:.0f} (deg: {rob['d2_deg']:.0%})")
    print(f"  Perturb: EV=${rob['perturb_ev']:.0f} (deg: {rob['perturb_deg']:.0%})")
    robust = rob["d1_deg"] < 0.30 and rob["d2_deg"] < 0.50 and rob["perturb_deg"] < 0.30
    print(f"  {'✓ ROBUST' if robust else '✗ NOT ROBUST'}")
    
    # ── Phase 4: Held-out with d=1 ──────────────────────────────────────
    print(f"\n--- PHASE 4: HELD-OUT (2007-2019, d=1) ---")
    df_old = load_nq_data(nq_path, year_start=2007)
    df_old = df_old[df_old.index < "2020-01-01"].copy()
    heldout = {"ev": 0, "chal_rate": 0, "fund_rate": 0, "live_profit": 0}
    if len(df_old) > 0:
        val_daily = generate_daily_pnl(df_old, bar, stop, sess, contracts=2, commission=1.50, entry_delay=1)
        heldout = compute_ev(val_daily, n_mc=N_MC)
        print(f"  EV=${heldout['ev']:.0f} ({heldout['chal_rate']:.0%}×{heldout['fund_rate']:.0%} "
              f"live=${heldout['live_profit']:.0f} n={len(val_daily)} days)")
        print(f"  {'✓ Profitable' if heldout['ev'] > 0 else '✗ Not profitable'}")
    
    # ── METRIC output ──────────────────────────────────────────────────
    print(f"METRIC ev_per_pipeline={rob['base_ev']:.2f}")
    print(f"METRIC robust_ev={rob['d1_ev']:.2f}")
    print(f"METRIC chal_pass_rate={rob['base_chal']:.4f}")
    print(f"METRIC fund_pass_rate={rob['base_fund']:.4f}")
    print(f"METRIC chal_fund_ratio={rob['base_chal']/rob['base_fund']:.4f}" if rob['base_fund'] > 0 else "")
    print(f"METRIC delay1_degradation={rob['d1_deg']:.4f}")
    print(f"METRIC delay2_degradation={rob['d2_deg']:.4f}")
    print(f"METRIC perturb_degradation={rob['perturb_deg']:.4f}")
    print(f"METRIC heldout_ev={heldout['ev']:.2f}")


if __name__ == "__main__":
    main()
