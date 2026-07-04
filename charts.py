"""
Execution Charts for VWAP Strategy
===================================
Generates visual reports: equity curve, trade entries, drawdown, PnL distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import time as dtime

from benchmark import load_nq_data, generate_daily_pnl, compute_ev


def generate_trade_details(
    df: pd.DataFrame,
    bar_minutes: int = 15,
    stop_atr_fraction: float = 1.0,
    session: str = "morning",
    contracts: int = 2,
    commission: float = 1.50,
    entry_delay: int = 1,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Generate detailed trade records for charting.
    Returns DataFrame with entry_time, exit_time, side, entry_price, exit_price, pnl, result.
    """
    from datetime import time as dtime

    if bar_minutes > 1:
        bars = df.resample(f"{bar_minutes}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    else:
        bars = df

    n = len(bars)
    if n < 100:
        return pd.DataFrame()

    # VWAP
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

    # ATR
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
    atr_daily = np.zeros(len(tr))
    atr_daily[0] = tr[0]
    alpha = 2.0 / (14 + 1)
    for i in range(1, len(tr)):
        atr_daily[i] = alpha * tr[i] + (1 - alpha) * atr_daily[i - 1]
    atr = atr_daily[group_id]

    # Session
    times = bars.index.time
    if session == "morning":
        session_mask = np.array([(dtime(9, 30) <= t <= dtime(12, 0)) for t in times])
    else:
        session_mask = np.array([(dtime(9, 30) <= t <= dtime(15, 45)) for t in times])
    eod_mask = np.array([t >= dtime(16, 0) for t in times])

    close = bars["close"].values.astype("float64")
    high = bars["high"].values.astype("float64")
    low = bars["low"].values.astype("float64")

    tick_size = 0.25
    tick_value = 5.0

    trades = []
    in_position = False
    side = 0
    entry_px = 0.0
    stop_px = 0.0
    entry_time = None
    pending_signal = 0
    signal_count = 0

    for i in range(n):
        cur_vwap = vwap[i]
        cur_atr = atr[i]
        ts = bars.index[i]

        if np.isnan(cur_vwap) or cur_vwap <= 0:
            continue

        if in_position and eod_mask[i]:
            pnl_ticks = (close[i] - entry_px) / tick_size * side
            pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
            trades.append({
                "entry_time": entry_time, "exit_time": ts, "side": "long" if side == 1 else "short",
                "entry_price": entry_px, "exit_price": close[i], "pnl": pnl_dollars,
                "result": "win" if pnl_dollars > 0 else "loss",
            })
            in_position = False
            continue

        if in_position:
            hit_stop = False
            exit_px = 0.0
            if side == 1 and low[i] <= stop_px:
                exit_px = stop_px
                hit_stop = True
            elif side == -1 and high[i] >= stop_px:
                exit_px = stop_px
                hit_stop = True
            if hit_stop:
                pnl_ticks = (exit_px - entry_px) / tick_size * side
                pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
                trades.append({
                    "entry_time": entry_time, "exit_time": ts, "side": "long" if side == 1 else "short",
                    "entry_price": entry_px, "exit_price": exit_px, "pnl": pnl_dollars,
                    "result": "win" if pnl_dollars > 0 else "loss",
                })
                in_position = False
            continue

        if not session_mask[i]:
            continue

        signal = 1 if close[i] > cur_vwap else (-1 if close[i] < cur_vwap else 0)
        if signal == 0:
            continue
        if long_only and signal == -1:
            continue

        if signal != pending_signal:
            pending_signal = signal
            signal_count = 0
            continue
        signal_count += 1
        if signal_count <= entry_delay:
            continue

        side = signal
        entry_px = close[i]
        entry_time = ts
        stop_dist = cur_atr * stop_atr_fraction if cur_atr > 0 else 10 * tick_size
        stop_px = entry_px - stop_dist * side
        in_position = True
        pending_signal = 0
        signal_count = 0

    if in_position:
        pnl_ticks = (close[-1] - entry_px) / tick_size * side
        pnl_dollars = pnl_ticks * tick_value * contracts - commission * 2 * contracts
        trades.append({
            "entry_time": entry_time, "exit_time": bars.index[-1], "side": "long" if side == 1 else "short",
            "entry_price": entry_px, "exit_price": close[-1], "pnl": pnl_dollars,
            "result": "win" if pnl_dollars > 0 else "loss",
        })

    return pd.DataFrame(trades)


def plot_execution_charts(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_pnl: pd.Series,
    output_dir: str | Path,
    title_prefix: str = "VWAP Strategy",
    show_days: int = 30,
):
    """Generate execution charts and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Equity Curve ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1, 1]})

    cum_pnl = daily_pnl.cumsum()
    equity = 100_000 + cum_pnl

    ax = axes[0]
    ax.plot(equity.index, equity.values, color="#2196F3", linewidth=1.2, label="Equity")
    ax.axhline(y=100_000, color="gray", linestyle="--", alpha=0.5, label="Start Balance")
    ax.fill_between(equity.index, 100_000, equity.values,
                    where=equity.values >= 100_000, alpha=0.15, color="green")
    ax.fill_between(equity.index, 100_000, equity.values,
                    where=equity.values < 100_000, alpha=0.15, color="red")
    ax.set_title(f"{title_prefix} — Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Account Value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 2. Daily PnL ──────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in daily_pnl.values]
    ax2.bar(daily_pnl.index, daily_pnl.values, color=colors, width=1.0, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_title("Daily PnL", fontsize=12)
    ax2.set_ylabel("PnL ($)")
    ax2.grid(True, alpha=0.3)

    # ── 3. Drawdown ───────────────────────────────────────────────────
    ax3 = axes[2]
    peak = equity.cummax()
    drawdown = equity - peak
    ax3.fill_between(drawdown.index, 0, drawdown.values, color="#F44336", alpha=0.4)
    ax3.set_title("Drawdown", fontsize=12)
    ax3.set_ylabel("Drawdown ($)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved equity_curve.png")

    # ── 4. Recent Trades on Price Chart ────────────────────────────────
    if len(trades_df) > 0:
        # Show last N days of trading
        recent_trades = trades_df.tail(min(50, len(trades_df)))
        start_date = recent_trades["entry_time"].min().normalize() - pd.Timedelta(days=2)
        end_date = recent_trades["exit_time"].max().normalize() + pd.Timedelta(days=2)

        df_window = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df_window) > 0:
            fig2, ax4 = plt.subplots(figsize=(18, 8))

            # Plot price
            ax4.plot(df_window.index, df_window["close"].values, color="#333", linewidth=0.8, alpha=0.8, label="NQ Close")

            # Plot trades
            for _, t in recent_trades.iterrows():
                color = "#4CAF50" if t["pnl"] > 0 else "#F44336"
                marker_entry = "^" if t["side"] == "long" else "v"
                ax4.scatter(t["entry_time"], t["entry_price"], marker=marker_entry,
                           color=color, s=80, zorder=5, edgecolors="black", linewidths=0.5)
                ax4.scatter(t["exit_time"], t["exit_price"], marker="x",
                           color=color, s=60, zorder=5)
                ax4.plot([t["entry_time"], t["exit_time"]],
                        [t["entry_price"], t["exit_price"]],
                        color=color, linewidth=1, alpha=0.5)

            ax4.set_title(f"{title_prefix} — Recent Trades (last {len(recent_trades)} trades)", fontsize=14, fontweight="bold")
            ax4.set_ylabel("NQ Price")
            ax4.set_xlabel("Date/Time")
            ax4.legend(loc="upper left")
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            plt.xticks(rotation=45)

            plt.tight_layout()
            fig2.savefig(output_dir / "recent_trades.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved recent_trades.png")

    # ── 5. PnL Distribution ───────────────────────────────────────────
    if len(trades_df) > 0:
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))

        # Trade PnL histogram
        ax5.hist(trades_df["pnl"], bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax5.axvline(x=0, color="red", linestyle="--")
        ax5.axvline(x=trades_df["pnl"].mean(), color="green", linestyle="--",
                    label=f"Mean: ${trades_df['pnl'].mean():.0f}")
        ax5.set_title("Trade PnL Distribution", fontsize=12)
        ax5.set_xlabel("PnL ($)")
        ax5.set_ylabel("Count")
        ax5.legend()

        # Win/loss pie
        wins = (trades_df["pnl"] > 0).sum()
        losses = (trades_df["pnl"] <= 0).sum()
        ax6.pie([wins, losses], labels=[f"Wins ({wins})", f"Losses ({losses})"],
                colors=["#4CAF50", "#F44336"], autopct="%1.1f%%", startangle=90)
        ax6.set_title(f"Win Rate: {wins/(wins+losses):.1%}", fontsize=12)

        plt.tight_layout()
        fig3.savefig(output_dir / "pnl_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved pnl_distribution.png")

    # ── 6. Monthly Returns Heatmap ─────────────────────────────────────
    if len(daily_pnl) > 60:
        monthly = daily_pnl.resample("ME").sum()
        monthly_pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "pnl": monthly.values,
        })
        heatmap_data = monthly_pivot.pivot_table(index="year", columns="month", values="pnl", fill_value=0)

        fig4, ax7 = plt.subplots(figsize=(14, max(4, len(heatmap_data) * 0.6)))
        im = ax7.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto")
        ax7.set_xticks(range(12))
        ax7.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax7.set_yticks(range(len(heatmap_data)))
        ax7.set_yticklabels(heatmap_data.index)
        for i in range(len(heatmap_data)):
            for j in range(12):
                val = heatmap_data.values[i, j]
                if val != 0:
                    ax7.text(j, i, f"${val:,.0f}", ha="center", va="center",
                            fontsize=8, color="black" if abs(val) < 5000 else "white")
        plt.colorbar(im, ax=ax7, label="Monthly PnL ($)")
        ax7.set_title(f"{title_prefix} — Monthly PnL Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig4.savefig(output_dir / "monthly_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"  Saved monthly_heatmap.png")

    print(f"\n  All charts saved to {output_dir}/")


def main():
    nq_path = str(PROJECT_ROOT / "data" / "raw" / "nq-1m.csv")
    output_dir = PROJECT_ROOT / "reports" / "execution_charts"

    # Best params: 15m, stop=1.0, morning, long_only, d=1
    bar, stop, sess, lo = 15, 1.0, "morning", True

    print("Loading data...")
    df = load_nq_data(nq_path, year_start=2020)
    print(f"  {len(df):,} bars loaded")

    print("Generating trades...")
    trades_df = generate_trade_details(df, bar, stop, sess, contracts=2, commission=1.50,
                                        entry_delay=1, long_only=lo)
    print(f"  {len(trades_df)} trades generated")

    print("Computing daily PnL...")
    daily = generate_daily_pnl(df, bar, stop, sess, contracts=2, commission=1.50,
                               entry_delay=1, long_only=lo)
    ev = compute_ev(daily)
    print(f"  EV=${ev['ev']:.0f} ({ev['chal_rate']:.0%}×{ev['fund_rate']:.0%}, {ev['expected_attempts']:.1f} attempts)")

    print("\nGenerating charts...")
    plot_execution_charts(df, trades_df, daily, output_dir,
                         title_prefix="VWAP Long-Only 15m Morning")

    # Trade summary
    if len(trades_df) > 0:
        print(f"\n{'='*50}")
        print(f"TRADE SUMMARY")
        print(f"{'='*50}")
        print(f"  EV per pipeline: ${ev['ev']:.0f}")
        print(f"  Expected attempts: {ev['expected_attempts']:.1f}")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Win rate: {(trades_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg win: ${trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean():,.0f}")
        print(f"  Avg loss: ${trades_df.loc[trades_df['pnl'] <= 0, 'pnl'].mean():,.0f}")
        print(f"  Total PnL: ${trades_df['pnl'].sum():,.0f}")
        print(f"  Best trade: ${trades_df['pnl'].max():,.0f}")
        print(f"  Worst trade: ${trades_df['pnl'].min():,.0f}")
        print(f"  Profit factor: {trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() / abs(trades_df.loc[trades_df['pnl'] <= 0, 'pnl'].sum()):.2f}" if (trades_df['pnl'] <= 0).any() else "")


if __name__ == "__main__":
    main()
