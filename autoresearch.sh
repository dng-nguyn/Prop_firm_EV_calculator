#!/usr/bin/env bash
# Autoresearch benchmark: VWAP Strategy EV on Prop Firm Calculator
# Runs parameter sweep, robustness tests, and held-out validation.
# Reports METRIC lines for autoresearch framework.

set -euo pipefail
cd "$(dirname "$0")"

python3 benchmark.py 2>/dev/null | grep -E '^METRIC '
