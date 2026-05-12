"""Newton determinism test harness.

Public entry points:

- :func:`scripts.run_determinism.main` — CLI driver.
- :data:`scripts.determinism.scenarios.SCENARIOS` — scenario registry.
- :data:`scripts.determinism.harness.SOLVER_SPECS` — solver registry.
"""

from __future__ import annotations

from .harness import (
    SOLVER_SPECS,
    Scenario,
    ScenarioArgs,
    ScenarioSnapshot,
    SolverSpec,
    compare_runs,
    hash_core,
    list_solvers,
    write_snapshot_to_path,
)
from .scenarios import SCENARIOS

__all__ = [
    "SCENARIOS",
    "SOLVER_SPECS",
    "Scenario",
    "ScenarioArgs",
    "ScenarioSnapshot",
    "SolverSpec",
    "compare_runs",
    "hash_core",
    "list_solvers",
    "write_snapshot_to_path",
]
