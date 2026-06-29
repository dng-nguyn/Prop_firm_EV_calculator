"""
Monte_carlo — Monte Carlo Random‑shuffle Simulation Engine

Takes the observed daily PnL sequence, shuffles it many times, and
evaluates the probability of passing a prop‑firm challenge.

Exports
-------
SimulationResult       — data‑class with aggregated statistics
run_simulations        — run N shuffled simulations
run_full_analysis      — run simulations + compute EV / expected cost
"""

from .simulator import (
    SimulationResult,
    run_full_analysis,
    run_simulations,
)

__all__ = [
    "SimulationResult",
    "run_full_analysis",
    "run_simulations",
]
