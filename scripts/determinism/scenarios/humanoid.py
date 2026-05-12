"""Scenario 5: Humanoid standing.

Loads Unitree H1 in a nominal standing pose with position targets held at
the initial joint configuration, so any drift/falling is solver-driven.
Solver robustness test — humanoid standing is famously unforgiving for
physics engines.

Extras captured:
  - ``com_history``: (samples, world_count, 3) center-of-mass trajectory
  - ``foot_contact_history``: (samples, world_count) rigid contact count
    (a proxy for foot contacts; sees *all* contacts but in the standing
    scene most contacts are feet-vs-ground)
  - ``fell_mask``: per-world bool -> did the root drop below HEIGHT_FALL_THRESHOLD
  - ``root_height_final``: per-world z of the root link at the last step
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode

from ..harness import Scenario


class HumanoidScenario(Scenario):
    id = "humanoid"
    supported_solvers = ("xpbd", "mujoco")

    HEIGHT_FALL_THRESHOLD = 0.6  # meters (root z): H1 nominal standing ≈ 1.0
    KP = 150.0
    KD = 5.0

    def _on_built(self) -> None:
        self._com_hist: list[np.ndarray] = []
        self._foot_contact_hist: list[np.ndarray] = []

        assert self.model is not None
        # Body count per world (for per-world slicing of body_q).
        self._bodies_per_world = int(self.model.body_count) // self.args.world_count

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        if self.args.solver.name == "mujoco":
            import newton.solvers as _s

            # Already registered by harness._configure_sub_builder, but
            # safe to call twice.
            _s.SolverMuJoCo.register_custom_attributes(builder)

        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5,
        )
        builder.default_shape_cfg.ke = 2.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd_structured" / "h1.usda")
        builder.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        # Collision cheapening (same recipe as example_robot_h1).
        builder.approximate_meshes("bounding_box")

        # Hold every joint at its configured initial q using position PD.
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = self.KP
            builder.joint_target_kd[i] = self.KD
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

    def per_step(self) -> None:
        if self.args.solver.name != "xpbd":
            return
        assert self.model is not None and self.state_0 is not None
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self) -> None:  # type: ignore[override]
        super().step()
        # Sample once per frame.
        body_q = self.state_0.body_q.numpy()
        per_world = body_q.reshape(self.args.world_count, self._bodies_per_world, 7)
        # COM proxy: average position of all bodies per world (cheap; the
        # actual mass-weighted COM would require reading model.body_mass).
        com = per_world[:, :, :3].mean(axis=1)
        self._com_hist.append(com.astype(np.float32))

        rc = int(self.contacts.rigid_contact_count.numpy()[0])
        # rigid_contact_count is a total over all worlds. For per-world
        # foot-contact trends we'd need to demux; report the total as a
        # robust coarse signal.
        self._foot_contact_hist.append(np.full(self.args.world_count, rc, dtype=np.int32))

    def extra_snapshot(self) -> dict[str, Any]:
        assert self.state_0 is not None
        body_q = self.state_0.body_q.numpy()
        per_world = body_q.reshape(self.args.world_count, self._bodies_per_world, 7)
        # Root is body 0 per world in H1's default import order.
        root_z = per_world[:, 0, 2]
        fell = root_z < self.HEIGHT_FALL_THRESHOLD

        com_hist = (
            np.stack(self._com_hist, axis=0)
            if self._com_hist else np.zeros((0, self.args.world_count, 3), dtype=np.float32)
        )
        foot_hist = (
            np.stack(self._foot_contact_hist, axis=0)
            if self._foot_contact_hist
            else np.zeros((0, self.args.world_count), dtype=np.int32)
        )

        return {
            "com_history": com_hist,
            "foot_contact_history": foot_hist,
            "fell_mask": fell.astype(np.uint8),
            "root_height_final": root_z.astype(np.float32),
            "bodies_per_world": self._bodies_per_world,
        }
