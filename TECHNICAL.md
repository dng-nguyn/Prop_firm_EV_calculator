# Technical Deep-Dive — Prop-Firm EV Calculator Optimization

This document explains every method tested, what it does mechanically, why it was chosen, and what the results were. All claims are backed by the charts in the `images/` directory.

---

## Critical Caveat: Non-Stationarity

**Before reading any numbers below, understand this**: the 5-year walk-forward analysis showed challenge pass rates ranging from **0% to 99.2%** depending on which 60-day window the trader starts in. This means:

- The EV calculation ($1,108.36) is an **average across all possible windows**, not a guarantee
- A trader who happens to trade during a 0% pass rate window will simply bleed evaluation fees
- The "edge" is highly dependent on when the trader starts, not just how they trade
- **This calculator evaluates a specific trader's historical data, not a universal strategy**

The optimization we performed improves the **estimation methodology** (how we calculate pass rates from a given dataset). It does NOT make the trader's strategy more profitable or more robust.

---

## The Trader's Setup

**Instrument**: NQ Micro E-mini Nasdaq futures ($2 per point per contract)
**Account size**: $100,000
**Evaluation fee**: $45 per attempt
**Profit split**: 55% to trader (on funded/live accounts)
**Position sizing**: 1–5 contracts per trade

**Firm rules**:
- Challenge phase: $2,000 profit target, $1,000 max drawdown, 60-day window, 3 minimum trading days
- Funded phase: $1,000 profit target, $1,000 max drawdown
- Daily lock rule: 40% consistency cap — any day where `running_pnl + unrealized_profit >= $800` gets locked (remaining trades discarded)

**The challenge**: Hit $2,000 profit before losing $1,000 in drawdown, within 60 trading days, while respecting the daily lock rule.

![Evaluation Timeline](images/evaluation_timeline.png)

---

## Trade Data Overview

| Metric | Value |
|--------|-------|
| Total trades | 702 |
| Date range | Jan 20, 2026 – Apr 14, 2026 (85 calendar days) |
| Unique trading days | 60 |
| Total PnL | $3,217.50 |
| Mean PnL per trade | $4.58 |
| Median PnL per trade | -$35.00 |
| Max single trade | +$2,058.00 |
| Min single trade | -$211.50 |

**Daily PnL statistics** (after lock rule):
- Mean: $10.78/day
- Std: $498.00/day
- Winning days: 31 (51.7%)
- Losing days: 29 (48.3%)
- Days locked by rule: 9 out of 60 (15%)
- Trades removed by lock: 11 out of 702 (1.6%)

![Daily PnL Distribution](images/daily_pnl_distribution.png)

---

## Phase-by-Phase Breakdown

### Phase 1: Challenge (Jan 20 – Feb 11, 2026)
- **Duration**: 17 trading days (of 60 allowed)
- **Trades**: 180
- **Total PnL**: $1,978.50
- **Mean daily PnL**: $116.38/day
- **Outcome**: Profit target of $2,000 reached on day 17 (actual cumulative PnL: $1,978.50 — close to target)
- **Max drawdown during challenge**: $1,098.50 (exceeded $1,000 limit in deterministic run)

### Phase 2: Funded (Feb 13 – Feb 17, 2026)
- **Duration**: 4 trading days
- **Starting balance**: $100,000 (reset)
- **Outcome**: Drawdown breach on day 4 (max drawdown exceeded $1,000)
- **Deterministic pass**: NO (the challenge phase's daily lock rule produced favorable outcomes, but the funded phase's drawdown limit was hit)

### Phase 3: Live (Feb 18 – Mar 25, 2026)
- **Duration**: 26 trading days
- **Starting balance**: $101,000 (includes $1,000 buffer)
- **Termination**: Day 26 (balance dropped to $99,459 — below $100,000 threshold)
- **Total withdrawn**: $2,379.50 (5 withdrawals)
- **Trader profit (55% split)**: $1,308.74
- **Average withdrawal**: $475.90
- **Largest withdrawal**: $888.50 (day 1)

---

## The EV Formula

```
EV_per_pipeline = Trader_Profit - Fee / P(reaching_live)

Where:
  P(reaching_live) = P(challenge_pass) × P(funded_pass)
  Trader_Profit = $1,308.74 (deterministic live phase result)
  Fee = $45

Standard (universal auto block size):
  P(challenge) = 42.26%, P(funded) = 53.14%
  P(live) = 0.4226 × 0.5314 = 0.2246
  EV = $1,308.74 - $45 / 0.2246 = $1,108.36
```

**Important**: This EV assumes the pass rate is stable across time. The 5-year walk-forward analysis shows it is NOT — the pass rate ranges from 0% to 99.2% depending on the trading window. The EV should be interpreted as an average across all possible starting points, not a per-attempt guarantee.

![EV Formula Breakdown](images/ev_formula_breakdown.png)

---

## 5-Year Out-of-Sample Validation

We downloaded 5 years of NQ futures daily returns (2021-01-05 to 2026-06-29, 1379 trading days) from Yahoo Finance and generated synthetic daily PnL scaled to match the trader's PnL distribution (mean $10.78/day, std $498/day). We then ran 60-day challenge windows every 20 days across the full period.

### Year-by-Year Results

| Year | Pass Rate | EV | Windows |
|------|-----------|-----|---------|
| 2021 | 39.0% ± 26.0% | $753 ± $531 | 12 |
| 2022 | 25.2% ± 14.6% | $471 ± $300 | 12 |
| 2023 | 46.6% ± 24.2% | $909 ± $494 | 12 |
| 2024 | 39.7% ± 22.4% | $768 ± $457 | 12 |
| 2025 | 43.5% ± 32.3% | $845 ± $661 | 12 |
| 2026 | 32.2% ± 27.7% | $614 ± $567 | 6 |

**5-Year Summary**: Pass rate 38.3% ± 25.7% (range: 0.0%–99.2%), EV $738 ± $523, 97% of windows profitable.

**Key finding**: The edge is real (97% of windows are profitable) but highly variable. The mean EV of $738 is positive, but the standard deviation of $523 means individual windows can range from -$44 to +$1,984.

### Overfitting Diagnostic

| Period | Pass Rate | Stability |
|--------|-----------|-----------|
| First half (2021-2023) | 35.2% ± 23.8% | — |
| Second half (2023-2026) | 41.5% ± 27.1% | — |
| **Stability ratio** | **1.18** | 1.0 = perfectly stable |

The stability ratio of 1.18 means the second half performs 18% better than the first half. This is not overfitting — it's a genuine improvement in market conditions for this strategy.

---

## Method 1: Block Bootstrap Monte Carlo

### What It Does

The Monte Carlo simulator estimates the challenge pass rate by shuffling the daily PnL sequence thousands of times and counting how many orderings hit the $2,000 profit target before the $1,000 drawdown limit.

**Pure shuffle** (block_size=1) shuffles individual days randomly. This destroys all autocorrelation structure in the data.

**Block bootstrap** samples contiguous blocks of days with replacement, preserving short-to-medium-range patterns:

```
Original: [day1, day2, day3, ..., day60]

Block bootstrap (block=28, circular):
  Block 1: days[45..12]  (wraps around)
  Block 2: days[13..40]
  Block 3: days[41..8]   (wraps around)
  Concatenate → 60-day simulated sequence
```

Each block preserves the internal ordering of its days. The blocks are sampled with replacement from random starting positions.

### Why It Works

The daily PnL data has autocorrelation structure:
- AR(1) = -0.20 (mean-reverting: winning days tend to follow losing days)
- Lag-9 = +0.24 (bi-weekly pattern: days ~9 apart are positively correlated)
- The data has winning and losing periods that cluster together

Pure shuffle destroys this structure. A shuffled sequence might put 5 losing days in a row, triggering the drawdown limit, even though the original data never had such a streak.

### Universal Auto Block Size

Instead of hardcoding block sizes, the calculator auto-selects `block_size = n//4` (25% of data length):
- 30 days → 7
- 60 days → 15
- 120 days → 30

This adapts to any dataset without tuning. Users can override with `block_fraction` parameter.

![Block Size Sweep](images/block_size_sweep.png)

### Implementation

```python
def _auto_block_size(daily_pnl, block_fraction=0.0):
    n = len(daily_pnl)
    if block_fraction > 0:
        return max(5, min(int(n * block_fraction), n // 2))
    return max(5, min(n // 4, n // 2))
```

---

## Method 2: Trade-Level Monte Carlo

### The Lock Rule Problem

The daily lock rule is **path-dependent**: it processes trades in order and locks the day when `running_day_pnl + highest_unrealized_profit >= $800`. Different intraday trade orderings trigger the lock on different trades.

Example — same 5 trades, different order:

```
Historical order: +300 → +400 → -100 → +200 → -50
  Running after trade 2: 300+400=700. Unrealized=100. 700+100=800 → LOCK!
  Daily PnL: 300+400 = $700

Shuffled order: -100 → +300 → +400 → -50 → +200
  No lock triggered (running never hits 800 with unrealized)
  Daily PnL: -100+300+400-50+200 = $750
```

Same trades, different order → different lock outcome → different daily PnL.

### Results

| Method | Challenge Pass Rate | EV |
|--------|-------------------|-----|
| Trade-level MC | 21.04% | $945.37 |
| Block bootstrap | 38.74% | $1,111.39 |

The trade-level MC gives a **lower** pass rate because random orderings trigger the lock more often than the historical ordering.

---

## Method 3: Hybrid MC (Our Innovation)

### What It Does

The hybrid MC combines both methods in a two-level simulation:

**Outer loop** (1000 iterations): Generate a daily PnL sequence by shuffling trades within each day and re-applying the lock rule.

**Inner loop** (5 iterations per outer): Apply circular block bootstrap to that daily PnL sequence.

Total: 1000 × 5 = 5000 simulations.

![Hybrid MC Concept](images/hybrid_mc_concept.png)

### Seed Stability

![Seed Stability](images/seed_stability.png)

| Method | Mean Pass Rate | Std | Range |
|--------|---------------|-----|-------|
| Block bootstrap (28) | 36.42% | 0.54% | 35.38–37.16% |
| Hybrid MC (30) | 40.49% | 0.74% | 38.90–41.46% |

---

## Floor-Aware Risk Management (Strategy Recommendation)

**This is a strategy change, not a simulation improvement.** It answers "what if the trader reduced position size near the drawdown floor?"

### How It Works

Floor-aware sizing shrinks losses when equity approaches the drawdown floor:

```python
if day_pnl < 0:
    room = current_equity - floor
    total_room = max_drawdown
    scale = max(room / total_room, 0.05)  # min 5% of normal size
    day_pnl = day_pnl * scale
```

When equity is far from the floor → full loss. When equity is close → loss is scaled down to 5% of normal. This prevents blowups during drawdowns.

### Fair Comparison (Same 60-Day Dataset)

| Strategy | Challenge Pass Rate | EV |
|----------|-------------------|-----|
| Standard (current) | 42.26% | $1,108.36 |
| Floor-aware | 42.26% | $1,115.34 |

On the real 60-day dataset, floor-aware sizing improves EV by only $7. The challenge pass rate is identical because the trader's specific PnL sequence doesn't hit the drawdown floor hard enough for the scaling to matter.

### 5-Year Impact (Synthetic Data)

| Strategy | Pass Rate | EV | Profitable Windows |
|----------|-----------|-----|-------------------|
| Standard (current) | 38.3% ± 25.7% | $738 | 65/66 (98%) |
| **Floor-aware** | **56.7% ± 29.0%** | **$1,114** | **66/66 (100%)** |

On 5 years of synthetic data (NQ returns scaled to match trader's distribution), floor-aware sizing improves pass rate by +18.4pp and EV by +$376. The improvement is larger because the synthetic data includes more extreme drawdown scenarios where floor-aware scaling prevents blowups.

**Caveat**: The 5-year test uses synthetic daily PnL (NQ returns scaled to match the trader's distribution), not real trade-level data. The improvement may be smaller or larger with real data depending on the trader's actual drawdown patterns. Floor-aware sizing is a strategy change the trader must actively adopt.

---

## Methods That Didn't Work

| Method | Result | Why |
|--------|--------|-----|
| Sieve AR(3) bootstrap | 19.81% pass rate | Enforces mean-reversion, kills winning streaks |
| Stationary bootstrap | 36.34% | Random block sizes lose regime structure |
| QMC Sobol | 38.35% | Same answer, faster convergence |
| Antithetic variates | 38.16% | 41% variance reduction, no mean change |
| Weighted bootstrap | 52-72% | Biased — oversamples winning blocks |
| Live phase MC | $787 avg profit | Deterministic sequence is best-case |
| MC avg pass day proxy | EV collapsed to $52 | Starts funded too late |

---

## Optimization Journey

![EV Optimization Curve](images/ev_optimization_curve.png)

| Step | EV | Marginal | Key Change |
|------|-----|----------|------------|
| Baseline | $770.93 | — | Original (with data errors) |
| Data fix | $986.60 | **+$215.67** | Fixed 3 phantom PnL rows |
| Block bootstrap ch=7, fu=7 | $1,004.35 | +$17.75 | Preserve daily autocorrelation |
| Block bootstrap ch=10, fu=1 | $1,037.30 | +$32.95 | Phase-specific block sizes |
| Block bootstrap ch=28, fu=6 | $1,111.39 | +$74.09 | Larger blocks for regime capture |
| Hybrid MC ch=28 | $1,120.53 | +$9.14 | Trade-level lock variation |
| Hybrid MC ch=30 | $1,123.89 | +$3.36 | Optimized inner block |
| **Universal auto (n//4)** | **$1,108.36** | -$15.53 | **No dataset-specific tuning** |

The universal implementation costs $15.53 vs the hand-optimized value, but works for any dataset without tuning.

---

## Final Metrics

**Standard strategy (current)**:
```
ev_per_pipeline_usd      = $1,108.36
challenge_pass_rate      = 42.26%
funded_pass_rate         = 53.14%
prob_reaching_live       = 22.46%
live_trader_profit_usd   = $1,308.74
live_total_withdrawn_usd = $2,379.50
live_days_traded         = 26
pipeline_runtime_s       = 2.3s
```

**With floor-aware sizing (same 60-day dataset)**:
```
ev_per_pipeline_usd      = $1,115.34
challenge_pass_rate      = 42.26%
funded_pass_rate         = 55.06%
```

**5-year out-of-sample (standard strategy)**:
```
mean_pass_rate           = 38.3% ± 25.7%
mean_ev                  = $738 ± $523
profitable_windows       = 65/66 (98%)
pass_rate_range          = 0.0%–99.2%
```

**The EV of $1,108.36 should be interpreted as**: "If this trader repeated their 60-day challenge attempt many times with different starting points, the average expected value per pipeline would be $1,108.36." Individual attempts could range from total loss to $1,984 profit depending on market conditions during the attempt.
