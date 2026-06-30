#!/usr/bin/env bash
# autoresearch.sh — Benchmark entrypoint for Prop-Firm EV Calculator optimization.
#
# Runs the full TraderLaunch evaluation pipeline with deterministic Monte Carlo
# seeds and reports key metrics.
#
# Primary metric: ev_per_pipeline_usd
# Exit 0 on success, non-zero on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run the benchmark
python3 benchmark.py
exit_code=$?

exit $exit_code
