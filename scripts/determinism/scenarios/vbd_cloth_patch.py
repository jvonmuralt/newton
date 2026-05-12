"""Small VBD cloth patch with pinned edge and ground contact."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from ..harness import Scenario, ScenarioSnapshot


class VbdClothPatchScenario(Scenario):
    """Pinned cloth patch evaluated with the VBD solver."""

    id = "vbd_cloth_patch"
    supported_solvers = ("vbd",)

    DIM_X = 10
    DIM_Y = 6
    CELL = 0.075

    def use_vbd_bending_coloring(self) -> bool:
        return True

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        builder.default_particle_radius = 0.01
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * self.DIM_X * self.CELL, -0.35, 1.1),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.25),
            vel=wp.vec3(0.2, 0.0, -0.2),
            dim_x=self.DIM_X,
            dim_y=self.DIM_Y,
            cell_x=self.CELL,
            cell_y=self.CELL,
            mass=0.02,
            fix_top=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=2.0e1,
            edge_kd=1.0e-1,
            particle_radius=0.01,
        )

    def snapshot(self) -> ScenarioSnapshot:
        assert self.state_0 is not None and self.model is not None
        particle_q = self.state_0.particle_q.numpy().copy()
        particle_qd = self.state_0.particle_qd.numpy().copy()
        particle_mass = self.model.particle_mass.numpy()
        core = {
            "particle_q": particle_q,
            "particle_qd": particle_qd,
        }
        extras = {
            "center_of_mass": particle_q.mean(axis=0).astype(np.float32),
            "height_range": np.asarray([particle_q[:, 2].min(), particle_q[:, 2].max()], dtype=np.float32),
            "pinned_particle_count": int(np.sum(particle_mass == 0.0)),
            "particle_count": int(self.model.particle_count),
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
            "collision_pipeline_deterministic": bool(getattr(self.model._collision_pipeline, "deterministic", False)),
            "collision_pipeline_wp_deterministic": self._collision_pipeline_warp_deterministic,
            "custom_collision_pipeline": self.collision_pipeline is not None,
            "graph_capture_enabled": self.graph is not None,
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)
