"""Scenario 2: Stack of 20 boxes — classic solver stress test.

The boxes are dropped from slightly perturbed heights so first-contact times
are staggered; this creates a deterministic but chaos-sensitive starting
condition that's a good determinism stress test.

Extras captured:
  - ``stack_order``: world 0 sort-by-height indices of the ``NUM_BOXES`` boxes
  - ``com_drift``: distance between the stack COM and the stack base XY
  - ``kinetic_energy``: sum of 0.5 * m * |v|^2 over all bodies
  - ``contact_count_history``: number of rigid contacts every 10 steps in world 0
"""

from __future__ import annotations

from typing import Any

import numpy as np

import newton

from ..harness import Scenario


class BoxStackScenario(Scenario):
    id = "box_stack"
    supported_solvers = ("xpbd", "vbd", "mujoco")

    NUM_BOXES = 20
    BOX_HALF = 0.1
    SPACING = 2.2 * BOX_HALF  # small gap so boxes settle rather than intersect
    LOG_EVERY = 10

    def _on_built(self) -> None:
        self._contact_history: list[int] = []

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.6

        for i in range(self.NUM_BOXES):
            # ``add_body`` auto-creates its own free joint + articulation.
            body = builder.add_body()
            builder.add_shape_box(body, hx=self.BOX_HALF, hy=self.BOX_HALF, hz=self.BOX_HALF)

            # Slight deterministic horizontal jitter — without any jitter the
            # stack is perfectly symmetric and no drift ever accumulates.
            # The jitter is reproducible via ``self.rng``.
            jitter_x = float(self.rng.uniform(-0.01, 0.01))
            jitter_y = float(self.rng.uniform(-0.01, 0.01))
            z = self.BOX_HALF + i * self.SPACING
            dof_start = i * 7
            builder.joint_q[dof_start : dof_start + 3] = [jitter_x, jitter_y, z]
            builder.joint_q[dof_start + 3 : dof_start + 7] = [0.0, 0.0, 0.0, 1.0]

    def step(self) -> None:  # type: ignore[override]
        super().step()
        # Log rigid contact counts outside the captured graph so the
        # ``.numpy()`` host copy doesn't serialize into the recorded CUDA
        # graph. Cheap because it's only every ``LOG_EVERY`` frames.
        if self.step_index % self.LOG_EVERY == 0:
            count = int(self.contacts.rigid_contact_count.numpy()[0])
            self._contact_history.append(count)

    def extra_snapshot(self) -> dict[str, Any]:
        assert self.state_0 is not None
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        # body_count == NUM_BOXES * world_count
        per_world = body_q.reshape(self.args.world_count, self.NUM_BOXES, 7)
        world0 = per_world[0]
        z_order = np.argsort(world0[:, 2])

        # Stack COM drift (world 0): distance from the expected stack base
        # column. Base column in world 0 is at (0, 0).
        com_xy = world0[:, :2].mean(axis=0)
        com_drift = float(np.linalg.norm(com_xy))

        # Crude KE proxy (mass=1 per box in defaults).
        v = body_qd[:, 3:6]
        w = body_qd[:, :3]
        kinetic_energy = float(0.5 * (np.sum(v * v) + np.sum(w * w)))

        return {
            "stack_order_world0": z_order.astype(np.int32),
            "com_drift_world0": com_drift,
            "kinetic_energy": kinetic_energy,
            "contact_count_history": np.asarray(self._contact_history, dtype=np.int32),
        }
