# Repository Guidelines

## Project Overview

Expected value (EV) calculator for proprietary trading firm challenges. Evaluates pass rates and EV of prop firm evaluation pipelines (challenge → funded → live) using deterministic EOD trailing drawdown analysis and Monte Carlo simulation on historical trade data. Trader Launch is fully implemented; Bulenox is scaffolded.

## Architecture & Data Flow

```
nq-1m.csv (data/raw/)              trades.csv (data/raw/)
  semicolon-delimited, no headers     comma-delimited, with headers
  OHLCV bars (date,time,O,H,L,C,V)    trade-level data
                                         │
  └── (external use)                    └── Firm Calculator
                                            ├── Step 1: Rules Filter Challenge
                                            ├── Step 2: Challenge Phase
                                            ├── Step 3: Rules Filter Funded
                                            ├── Step 4: Funded Phase
                                            ├── Step 5: Live Phase (Trader Launch only)
                                            └── Step 6: Final Result
                                                  └── reports/results/*.txt + *.yaml
                                                      reports/figures_png/*.png
```

### Data Formats — Read This First

There are **two distinct CSV files** in `data/raw/` that serve different purposes:

| File | Format | Purpose |
|------|--------|---------|
| `nq-1m.csv` | Semicolon-delimited, **no headers** | Raw 1-minute OHLCV bars (`date,time,open,high,low,close,volume`). Used externally; **not** consumed by the calculator. |
| `trades.csv` | Comma-delimited, **with headers** | Trade-level data consumed by `load_trades()`. This is the file the pipeline needs. |

**Column mismatch warning**: The README lists `entry_time, exit_time, entry_price, exit_price, quantity, pnl, highest_unrealized_profit`, but `load_trades()` (`Trader_launch_calculator.py:108-131`) expects `entry_time, exit_time, side, entry_price, exit_price, quantity, pnl, duration_min`. Downstream modules (Challenge Phase) also use `highest_unrealized_profit` for the consistency rule's capped-profit calculation. When building `trades.csv`, include **all columns** from both lists to avoid runtime `KeyError`s.

Shared engines in `Common/` provide the core calculations:
- **EOD Trailing Drawdown** (`Common/EOD/trailing_drawdown.py`): Three-tier API — `simulate_pnl_sequence` (low-level numpy), `simulate` (mid-level DataFrame), `analyze` (high-level dict). Implements floor-based trailing drawdown with floor locking.
- **Monte Carlo Simulator** (`Common/Monte_carlo/simulator.py`): Shuffles daily PnL N times, runs EOD simulation per shuffle, returns pass/fail rates and EV stats. Depends on `Common.EOD`.
- **Consistency Rule** (`Common/rules/Consistancy/__init__.py`): Checks the 40% rule — max profitable day must not exceed 40% of gross profits.

## Key Directories

| Path | Purpose |
|------|---------|
| `Prop_firm/Trader_launch/` | Fully implemented firm calculator (6-step pipeline) |
| `Prop_firm/Trader_launch/calculator_logic/` | Pipeline step modules (rules filters, phase calculators, final result) |
| `Prop_firm/Trader_launch/rules/` | YAML config + human-readable rule notes |
| `Prop_firm/Trader_launch/reports/` | Generated text reports and equity curve PNGs |
| `Prop_firm/Bulenox/` | Scaffolded firm calculator (all stubs, no implementation) |
| `Common/EOD/` | End-of-day trailing drawdown engine |
| `Common/Monte_carlo/` | Monte Carlo simulation engine |
| `Common/rules/` | Shared rule checkers (consistency) |
| `Common/Intraday/`, `Common/Static/` | Placeholder modules for future drawdown types |
| `data/raw/` | Raw NQ 1-minute futures OHLCV data (nq-1m.csv) |
| `data/processed_data/` | Empty — for cleaned/transformed trade data |

## Development Commands

```bash
# Run the Trader Launch calculator
python Prop_firm/Trader_launch/Trader_launch_calculator.py

# No test suite exists
# No build system, package manager, or CI/CD configured
```

## Code Conventions & Common Patterns

### Naming
- **snake_case** for functions, variables, file names: `aggregate_daily_pnl()`, `trailing_drawdown.py`
- **PascalCase** for dataclasses: `EODSnapshot`, `EODResult`, `SimulationResult`, `LiveResult`, `Withdrawal`
- **UPPER_CASE** for constants: `START_BALANCE`, `PROFIT_SPLIT`, `PROJECT_ROOT`, `RULES_YAML`
- **Leading underscore** for private helpers: `_load_rules()`, `_generate_plot()`
- Directory names use PascalCase: `Trader_launch/`, `Common/`, `Monte_carlo/`

### Dataclasses
All structured results use `@dataclass` with `from __future__ import annotations`. Key types:
- `EODSnapshot` — per-day state (equity, floor, PnL, floor_locked flag)
- `EODResult` — complete EOD analysis (passed/breached, snapshots, series accessors)
- `SimulationResult` — MC aggregated stats (pass_rate, avg days, run_details)
- `LiveResult` / `Withdrawal` — live phase simulation output

### Calculator Pattern
Each firm follows an identical skeleton:
```
Prop_firm/<Firm>/
├── <Firm>_calculator.py           # Entry point: path constants, run_pipeline(), step_runner(), load_trades()
├── calculator_logic/
│   ├── <Firm>_rules_filter_Challenge_phase.py  # Pre-checks (YAML-driven)
│   ├── Challenge_phase.py                       # run(trades, config, reports_dir, figures_dir, ...) -> dict
│   ├── <Firm>_rules_filter_Funded_phase.py
│   ├── Funded_phase.py
│   ├── Live_phase.py              # Trader Launch only
│   └── Final_result.py            # Aggregates all phases into EV metrics
├── rules/
│   └── rule_sets_metadata.yaml    # Machine-readable config (fees, drawdown limits, targets, contract limits)
└── reports/
    ├── results/                   # *.txt (human-readable) + Final_result_metadata.yaml (machine-readable)
    └── figures_png/               # Equity curve PNGs
```

Each calculator_logic module exposes a `run()` function returning a dict. The pipeline orchestrator (`step_runner`) wraps each call with logging and error handling.

### Import Pattern
Each `calculator_logic/` module inserts the project root into `sys.path` at import time (`sys.path.insert(0, str(_PROJECT_ROOT))`) so that `from Common.xxx import ...` works regardless of the working directory. This is the only way `Common` is reachable — there is no package install or `__init__.py` at the project root.

### Configuration
- **YAML** (`rule_sets_metadata.yaml`): Machine-readable numeric config loaded by code. Structure: `general.*`, `challenge.*`, `funded.*`.
- **TXT** (`rule_sets.txt`, `rule_sets_payout.txt`, etc.): Human-readable documentation only — not parsed by code.

### Error Handling
- Graceful import degradation: each calculator_logic import wrapped in try/except, module set to `None` on failure, step skipped at runtime.
- `RuntimeError` raised when rules filter pre-checks fail.
- Edge cases explicitly handled: empty PnL sequences, missing columns, division by zero (MC pass_rate=0 → `float('inf')`), no trades in live phase.

### Logging
Module-level `log = logging.getLogger(__name__)` pattern throughout. `.warning()` for non-fatal issues.

## Important Files

| File | Role |
|------|------|
| `Prop_firm/Trader_launch/Trader_launch_calculator.py` | **Primary entry point** — CLI `main()`, pipeline orchestrator |
| `Common/EOD/trailing_drawdown.py` | Core engine — EOD floor-based trailing drawdown |
| `Common/Monte_carlo/simulator.py` | Monte Carlo pass-rate estimation |
| `Common/rules/Consistancy/__init__.py` | 40% consistency rule checker |
| `Prop_firm/Trader_launch/rules/rule_sets_metadata.yaml` | Trader Launch rule configuration |
| `architecture.txt` | Project structure documentation |
| `todo.txt` | Outstanding work items |
| `main.py` | Top-level entry point (empty — not implemented) |
| `gatherAllFirmsResults.py` | Cross-firm aggregation (stub) |

## Runtime/Tooling Preferences

- **Python 3.10+** required (`X | Y` union syntax, `from __future__ import annotations`)
- **No package manager** — no requirements.txt, pyproject.toml, setup.py, or Pipfile
- **External dependencies**: `pandas`, `numpy`, `matplotlib`, `PyYAML`
- **Stdlib**: `logging`, `dataclasses`, `typing`, `pathlib`
- **No virtual environment** configured — run directly with system Python
- **No linting/formatting** tools configured (no ruff, flake8, black, isort)

## Testing & QA

**No test suite exists.** No pytest, unittest, or any test files. No CI/CD. Verification is manual — run the calculator and inspect generated reports.

## Known TODOs (from todo.txt)

1. Fix consistency rule calculation (currently incorrect)
2. Separate inactivity rules properly
3. Add min/max lot size separation
4. Add "stop after challenge pass" logic
