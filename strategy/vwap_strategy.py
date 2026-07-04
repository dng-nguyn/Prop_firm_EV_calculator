"""
VWAP-based Strategy Generator for NQ Futures
=============================================
Generates trades from 1-min OHLCV data using VWAP as the trend filter.
Based on Zarattini & Aziz (2023): long above VWAP, short below VWAP.

Key robustness features:
- Entry delay parameter (simulate execution lag)
- ATR-based stops (from ORB paper: tight stops, let profits run)
- Session filter (most VWAP profits come from 9:30-12:00 and 15:00-16:00 ET)
- Volatility targeting (Harvey et al. 2018: scale position by inverse vol)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# NQ/MNQ tick specs (per rule_sets.txt: "$0.50 per contract per side")
TICK_SIZE = 0.25
MNQ_TICK_VALUE = 1.25    # $1.25 per tick for MNQ (micro NQ)
NQ_TICK_VALUE = 5.0       # $5.00 per tick for full NQ
COMMISSION_PER_SIDE = 0.50  # $0.50/contract/side per rules


@dataclass
class StrategyParams:
    """Configurable strategy parameters."""
    # VWAP
    vwap_include_premarket: bool = False
    
    # Entry
    entry_delay_bars: int = 1       # Wait N bars after signal before entering
    session_start: str = "09:31"    # NY time, first valid entry
    session_end: str = "15:45"      # NY time, last entry (exit at 16:00)
    session_filter: str = "full"    # "full", "morning" (9:30-12:00), "afternoon" (13:00-16:00)
    
    # Stop loss
    stop_method: str = "atr"        # "atr", "fixed_ticks", "vwap_cross"
    stop_atr_fraction: float = 0.05 # Fraction of 14-day ATR (ORB paper optimal)
    stop_fixed_ticks: int = 20      # Fixed stop in ticks
    
    # Profit target
    target_method: str = "eod"      # "eod", "atr_multiple", "fixed_ticks"
    target_atr_multiple: float = 10.0
    target_fixed_ticks: int = 100
    
    # Position sizing
    contracts: int = 1
    risk_per_trade_pct: float = 1.0 # % of account to risk per trade
    
    # Volatility targeting (Harvey et al. 2018)
    vol_target_enabled: bool = False
    vol_target_half_life: int = 20  # Days, EWMA half-life
    vol_target_annual: float = 0.10 # Target annualized vol
    
    # Commissions
    commission_per_side: float = 0.50  # Per contract per side (per rules)


@dataclass
class Trade:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str                       # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    duration_min: float
    highest_unrealized_profit: float


def load_nq_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    df.sort_index(inplace=True)
    return df


def resample_to_bars(df: pd.DataFrame, minutes: int = 1) -> pd.DataFrame:
    """Resample 1-min data to N-minute bars."""
    if minutes <= 1:
        return df
    rule = f"{minutes}min"
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


def compute_intraday_vwap(df: pd.DataFrame, include_premarket: bool = False) -> pd.Series:
    """
    Compute intraday VWAP, resetting each trading day.
    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (H + L + C) / 3
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    
    # Group by trading date
    dates = df.index.date
    
    cum_tp_vol = (typical * vol).groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    vwap = vwap.ffill()
    
    return vwap


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR using daily bars resampled from intraday data."""
    daily = df.resample("1D").agg({
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    
    high = daily["high"]
    low = daily["low"]
    close = daily["close"].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    
    # Map back to intraday timestamps
    atr_intraday = atr.reindex(df.index, method="ffill")
    return atr_intraday


def compute_realized_vol(df: pd.DataFrame, half_life: int = 20) -> pd.Series:
    """Compute realized volatility using EWMA (for vol targeting)."""
    # Daily returns from close prices
    daily_close = df["close"].resample("1D").last().dropna()
    returns = daily_close.pct_change()
    
    # EWMA variance
    var = returns.ewm(span=half_life, adjust=False).var()
    vol = np.sqrt(var) * np.sqrt(252)  # Annualized
    
    # Map back to intraday
    vol_intraday = vol.reindex(df.index, method="ffill")
    return vol_intraday


def is_rth(ts: pd.Timestamp) -> bool:
    """Check if timestamp is in Regular Trading Hours (9:30-16:00 ET)."""
    t = ts.time()
    from datetime import time
    return time(9, 30) <= t <= time(16, 0)


def is_session_start(ts: pd.Timestamp, session: str = "full") -> bool:
    """Check if timestamp is within the allowed session window."""
    t = ts.time()
    from datetime import time
    if session == "morning":
        return time(9, 30) <= t <= time(12, 0)
    elif session == "afternoon":
        return time(13, 0) <= t <= time(16, 0)
    else:  # full
        return time(9, 30) <= t <= time(15, 45)


def generate_trades(
    df: pd.DataFrame,
    params: StrategyParams,
    bar_minutes: int = 1,
) -> list[Trade]:
    """
    Generate trades from OHLCV data using VWAP strategy.
    
    Logic (from Zarattini & Aziz 2023):
    - Long when close > VWAP (within RTH session window)
    - Short when close < VWAP
    - Exit: at EOD (16:00) or when stop hit
    - Entry delayed by `entry_delay_bars` to simulate execution lag
    """
    # Resample if needed
    bars = resample_to_bars(df, bar_minutes)
    
    # Compute indicators
    vwap = compute_intraday_vwap(bars, params.vwap_include_premarket)
    atr = compute_atr(bars)
    
    # Vol targeting
    vol_scale = None
    if params.vol_target_enabled:
        realized_vol = compute_realized_vol(bars, params.vol_target_half_life)
        vol_scale = params.vol_target_annual / realized_vol.replace(0, np.nan)
        vol_scale = vol_scale.clip(0.2, 3.0)  # Cap leverage
    
    trades: list[Trade] = []
    
    # Track state
    in_position = False
    position_side = None
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0
    highest_unreal = 0.0
    signal_bar_idx = None
    pending_signal = None
    entry_bar_count = 0
    
    dates = bars.index.date
    unique_dates = sorted(set(dates))
    
    for date in unique_dates:
        day_mask = dates == date
        day_bars = bars[day_mask]
        day_vwap = vwap[day_mask]
        day_atr = atr[day_mask]
        
        if len(day_bars) < 2:
            continue
        
        # Reset at start of each day
        in_position = False
        pending_signal = None
        entry_bar_count = 0
        
        for i, (ts, row) in enumerate(day_bars.iterrows()):
            cur_vwap = day_vwap.iloc[i]
            cur_atr = day_atr.iloc[i]
            
            if pd.isna(cur_vwap) or cur_vwap <= 0:
                continue
            
            # EOD exit: close all positions at 16:00
            from datetime import time
            if ts.time() >= time(16, 0) and in_position:
                exit_price = row["close"]
                pnl_ticks = (exit_price - entry_price) / TICK_SIZE
                if position_side == "short":
                    pnl_ticks = -pnl_ticks
                pnl_dollars = pnl_ticks * MNQ_TICK_VALUE * params.contracts
                pnl_dollars -= params.commission_per_side * 2 * params.contracts
                
                duration = (ts - entry_time).total_seconds() / 60.0
                
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    side=position_side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=params.contracts,
                    pnl=pnl_dollars,
                    duration_min=duration,
                    highest_unrealized_profit=highest_unreal,
                ))
                in_position = False
                continue
            
            # If in position, check stop
            if in_position:
                # Update highest unrealized profit
                if position_side == "long":
                    unreal = (row["high"] - entry_price) / TICK_SIZE * MNQ_TICK_VALUE * params.contracts
                else:
                    unreal = (entry_price - row["low"]) / TICK_SIZE * MNQ_TICK_VALUE * params.contracts
                highest_unreal = max(highest_unreal, unreal)
                
                # Check stop
                stopped = False
                if position_side == "long" and row["low"] <= stop_price:
                    exit_price = stop_price
                    stopped = True
                elif position_side == "short" and row["high"] >= stop_price:
                    exit_price = stop_price
                    stopped = True
                
                if stopped:
                    pnl_ticks = (exit_price - entry_price) / TICK_SIZE
                    if position_side == "short":
                        pnl_ticks = -pnl_ticks
                    pnl_dollars = pnl_ticks * MNQ_TICK_VALUE * params.contracts
                    pnl_dollars -= params.commission_per_side * 2 * params.contracts
                    
                    duration = (ts - entry_time).total_seconds() / 60.0
                    
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=ts,
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=params.contracts,
                        pnl=pnl_dollars,
                        duration_min=duration,
                        highest_unrealized_profit=highest_unreal,
                    ))
                    in_position = False
                continue
            
            # Not in position — check for signal
            if not is_session_start(ts, params.session_filter):
                continue
            
            # Determine signal
            signal = None
            if row["close"] > cur_vwap:
                signal = "long"
            elif row["close"] < cur_vwap:
                signal = "short"
            
            if signal is None:
                continue
            
            # Entry delay: wait N bars after signal
            if pending_signal != signal:
                pending_signal = signal
                entry_bar_count = 0
                continue
            
            entry_bar_count += 1
            if entry_bar_count < params.entry_delay_bars:
                continue
            
            # Enter position
            position_side = signal
            entry_price = row["close"]
            entry_time = ts
            
            # Set stop
            if params.stop_method == "atr":
                if cur_atr > 0:
                    stop_distance = cur_atr * params.stop_atr_fraction
                else:
                    stop_distance = 10 * TICK_SIZE  # Fallback: 10 ticks
            elif params.stop_method == "fixed_ticks":
                stop_distance = params.stop_fixed_ticks * TICK_SIZE
            else:  # vwap_cross
                stop_distance = abs(entry_price - cur_vwap)
                if stop_distance < 2 * TICK_SIZE:
                    stop_distance = 2 * TICK_SIZE
            
            if position_side == "long":
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
            
            in_position = True
            highest_unreal = 0.0
            pending_signal = None
            entry_bar_count = 0
    
    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert trades to DataFrame matching calculator expected format."""
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "side", "entry_price", "exit_price",
            "quantity", "pnl", "duration_min", "highest_unrealized_profit",
        ])
    
    records = []
    for t in trades:
        records.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "duration_min": t.duration_min,
            "highest_unrealized_profit": t.highest_unrealized_profit,
        })
    
    return pd.DataFrame(records)


def run_strategy(
    nq_csv_path: str | Path,
    params: StrategyParams | None = None,
    bar_minutes: int = 1,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load data → generate trades → return DataFrame.
    
    Parameters
    ----------
    nq_csv_path : path to nq-1m.csv
    params : strategy parameters (defaults if None)
    bar_minutes : resample to N-minute bars before applying strategy
    output_csv : if set, write trades.csv here
    
    Returns
    -------
    DataFrame of trades
    """
    if params is None:
        params = StrategyParams()
    
    log.info("Loading NQ data from %s", nq_csv_path)
    df = load_nq_data(nq_csv_path)
    log.info("  → %d bars loaded (%s to %s)", len(df), df.index[0], df.index[-1])
    
    log.info("Generating trades with bar_minutes=%d", bar_minutes)
    trades = generate_trades(df, params, bar_minutes)
    log.info("  → %d trades generated", len(trades))
    
    trade_df = trades_to_dataframe(trades)
    
    if output_csv:
        trade_df.to_csv(output_csv, index=False)
        log.info("  → written to %s", output_csv)
    
    return trade_df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    nq_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "nq-1m.csv"
    out_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "trades.csv"
    
    params = StrategyParams()
    
    # Allow CLI overrides
    if len(sys.argv) > 1:
        params.entry_delay_bars = int(sys.argv[1])
    if len(sys.argv) > 2:
        bar_minutes = int(sys.argv[2])
    else:
        bar_minutes = 5  # Default: 5-min bars (paper uses 1-min but 5-min more robust)
    
    trades_df = run_strategy(nq_path, params, bar_minutes, out_path)
    
    # Print summary
    if len(trades_df) > 0:
        print(f"\nStrategy Summary:")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Total PnL: ${trades_df['pnl'].sum():,.2f}")
        print(f"  Avg PnL/trade: ${trades_df['pnl'].mean():,.2f}")
        print(f"  Win rate: {(trades_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg win: ${trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean():,.2f}" if (trades_df['pnl'] > 0).any() else "  Avg win: N/A")
        print(f"  Avg loss: ${trades_df.loc[trades_df['pnl'] <= 0, 'pnl'].mean():,.2f}" if (trades_df['pnl'] <= 0).any() else "  Avg loss: N/A")
        print(f"  Max win: ${trades_df['pnl'].max():,.2f}")
        print(f"  Max loss: ${trades_df['pnl'].min():,.2f}")
    else:
        print("No trades generated.")
