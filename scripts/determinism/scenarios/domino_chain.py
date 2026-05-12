"""Scenario 3: Domino chain.

A row of tall thin boxes; the first domino starts tilted so the chain
topples deterministically. Extremely sensitive to numerical drift because
each domino's impact on the next depends on precise contact timing.

Extras captured:
  - ``fallen_count``: how many dominos have tipped past 45 degrees
  - ``propagation_time``: first step at which the *last* domino was tipped,
    in steps; -1 if it never fell
  - ``first_domino_angle``: tilt angle of domino 0 (for smoke diagnostics)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario


class DominoChainScenario(Scenario):
    id = "domino_chain"
    supported_solvers = ("xpbd", "mujoco")

    NUM_DOMINOS = 12
    DOMINO_HX = 0.02  # thin
    DOMINO_HY = 0.08
    DOMINO_HZ = 0.18  # tall
    SPACING_Z = 0.12  # closer than DOMINO_HZ to guarantee impact
    INITIAL_TILT_RAD = 0.25  # 14 degrees

    def _on_built(self) -> None:
        self._fallen_first_step: dict[int, int] = {}

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.6

        for i in range(self.NUM_DOMINOS):
            # ``add_body`` auto-creates its own free joint + articulation.
            body = builder.add_body()
            builder.add_shape_box(
                body, hx=self.DOMINO_HX, hy=self.DOMINO_HY, hz=self.DOMINO_HZ
            )

            # First domino gets a forward tilt so it tips toward +X.
            tilt = self.INITIAL_TILT_RAD if i == 0 else 0.0
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)

            # Place so the base sits on the ground with a small safety margin.
            x = i * self.SPACING_Z
            z = self.DOMINO_HZ + 1e-3
            dof = i * 7
            builder.joint_q[dof : dof + 3] = [x, 0.0, z]
            builder.joint_q[dof + 3 : dof + 7] = [q[0], q[1], q[2], q[3]]

    def step(self) -> None:  # type: ignore[override]
        super().step()
        # Track which dominos have tipped past 45 degrees from upright by
        # checking their local Z axis (body up vector).
        body_q = self.state_0.body_q.numpy()
        per_world = body_q.reshape(self.args.world_count, self.NUM_DOMINOS, 7)
        world0 = per_world[0]
        # Quaternion xyzw -> rotated +Z axis z-component = 1 - 2*(x^2+y^2).
        qx, qy = world0[:, 3], world0[:, 4]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        fallen = np.where(up_z < np.cos(np.pi / 4))[0]
        for idx in fallen:
            self._fallen_first_step.setdefault(int(idx), self.step_index)

    def extra_snapshot(self) -> dict[str, Any]:
        assert self.state_0 is not None
        body_q = self.state_0.body_q.numpy()
        per_world = body_q.reshape(self.args.world_count, self.NUM_DOMINOS, 7)
        world0 = per_world[0]

        qx, qy = world0[:, 3], world0[:, 4]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        fallen_count = int(np.sum(up_z < np.cos(np.pi / 4)))

        last_idx = self.NUM_DOMINOS - 1
        propagation_time = self._fallen_first_step.get(last_idx, -1)

        # Angle of domino 0 from upright.
        first_up_z = float(up_z[0])
        first_angle = float(np.degrees(np.arccos(np.clip(first_up_z, -1.0, 1.0))))

        return {
            "fallen_count_world0": fallen_count,
            "propagation_time_steps": propagation_time,
            "first_domino_angle_deg": first_angle,
            "fallen_step_map": dict(self._fallen_first_step),
        }
