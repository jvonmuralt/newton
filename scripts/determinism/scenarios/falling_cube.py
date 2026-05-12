"""Scenario 1: Falling cube.

One rigid cube per world, dropped from a small height onto the ground plane.
Simple but surprisingly useful for shaking out floor-contact determinism
because any rounding drift accumulates over ``num_steps`` substeps.

Extras captured:
  - ``final_pose``: body_q of the cube per world
  - ``final_velocity``: body_qd of the cube per world
  - ``final_height``: center-of-mass z coordinate per world
  - ``awake_count``: how many cubes are still moving (|v| > threshold) per world
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario


class FallingCubeScenario(Scenario):
    id = "falling_cube"
    supported_solvers = ("xpbd", "featherstone", "semi_implicit", "vbd", "mujoco")

    CUBE_HALF = 0.15
    DROP_HEIGHT = 1.5
    VELOCITY_SLEEP_THRESHOLD = 1e-3

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.5

        # ``add_body`` already auto-creates a free joint + articulation.
        # Do NOT also call ``add_joint_free`` (would produce 2 free joints
        # and 12 joint_qd DOFs per body).
        cube = builder.add_body()
        builder.add_shape_box(
            cube,
            hx=self.CUBE_HALF,
            hy=self.CUBE_HALF,
            hz=self.CUBE_HALF,
        )

        # Drop with a small deterministic rotation so the cube lands on an
        # edge rather than perfectly flat — this magnifies inter-run drift
        # in a reproducible way (from ``self.rng``, never the global RNG).
        angle = float(self.rng.uniform(-0.2, 0.2))
        axis = wp.normalize(wp.vec3(1.0, 0.5, 0.2))
        q = wp.quat_from_axis_angle(axis, angle)
        builder.joint_q[:3] = [0.0, 0.0, self.DROP_HEIGHT]
        builder.joint_q[3:7] = [q[0], q[1], q[2], q[3]]

    def extra_snapshot(self) -> dict[str, Any]:
        assert self.model is not None and self.state_0 is not None
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        # body_count == world_count here (one body per world).
        final_height = body_q[:, 2].copy()
        speed = np.linalg.norm(body_qd[:, 3:6], axis=1)
        awake = int(np.sum(speed > self.VELOCITY_SLEEP_THRESHOLD))

        return {
            "final_pose": body_q.copy(),
            "final_velocity": body_qd.copy(),
            "final_height": final_height,
            "awake_count": awake,
            "mean_speed": float(speed.mean()),
        }
