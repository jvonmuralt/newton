"""Shared helpers for small Warp deterministic-lowering scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ..harness import Scenario, ScenarioSnapshot

MICRO_SOLVERS = ("xpbd",)


class WarpMicroScenario(Scenario):
    """Base class for non-physics Warp lowering workloads."""

    supported_solvers = MICRO_SOLVERS
    graph_capture_enabled = True
    expected_current_gap = ""

    def build_subworld(self, builder: Any) -> None:
        del builder

    def build(self, viewer: Any) -> None:
        if self.args.viewer_name != "null":
            raise ValueError(f"{self.id} is a headless lowering scenario; use --viewer null.")

        self.device = wp.get_device()
        self._build_workload()

        if self.graph_capture_enabled and self.device.is_cuda:
            with wp.ScopedCapture() as cap:
                self._run_workload()
            self.graph = cap.graph
        else:
            self.graph = None

        # ViewerNull does not require a model.
        self.viewer = viewer

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._run_workload()
        self.step_index += 1

    def snapshot(self) -> ScenarioSnapshot:
        core = self._core_arrays()
        extras = {
            "description": self.__doc__.strip() if self.__doc__ else "",
            "expected_current_gap": self.expected_current_gap,
        }
        meta = {
            "scenario": self.id,
            "solver": self.args.solver.name,
            "world_count": self.args.world_count,
            "num_steps": self.args.num_steps,
            "seed": self.args.seed,
            "fps": self.args.fps,
            "substeps": self.args.substeps,
            "warp_version": wp.__version__,
            "wp_deterministic": wp.config.deterministic,
            "graph_capture_enabled": self.graph is not None,
            "micro_scenario": True,
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)

    def _build_workload(self) -> None:
        raise NotImplementedError

    def _run_workload(self) -> None:
        raise NotImplementedError

    def _core_arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError
