"""Scenario 4: 7-DOF arm PD tracking.

Loads Franka Panda (FR3) via URDF, fixes its base, and applies a
sinusoidal position target to each of the 7 arm DOFs via a PD torque
controller computed in a Warp kernel. Captures the joint state plus a
torque time series sampled once per frame.

Extras captured:
  - ``joint_q_history``: shape (samples, world_count, 7)
  - ``joint_qd_history``: shape (samples, world_count, 7)
  - ``joint_tau_history``: shape (samples, world_count, 7)
  - ``final_target_error``: max |q - target| per world at the last step
  - ``contact_count_history``: rigid contact counts sampled at frame boundaries
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton
import newton.utils

from ..harness import Scenario

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _pd_control(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    kp: wp.array[wp.float32],
    kd: wp.array[wp.float32],
    tau_limit: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
):
    """PD torque: tau = clamp(kp * (target - q) - kd * qd, -lim, lim)."""
    dof = wp.tid()
    err = target_q[dof] - joint_q[dof]
    tau_raw = kp[dof] * err - kd[dof] * joint_qd[dof]
    lim = tau_limit[dof]
    joint_f[dof] = wp.clamp(tau_raw, -lim, lim)


@wp.kernel
def _write_targets(
    t: float,
    amplitude: wp.array[wp.float32],
    phase: wp.array[wp.float32],
    bias: wp.array[wp.float32],
    freq: float,
    # outputs
    target_q: wp.array[wp.float32],
):
    """target[i] = bias[i] + amplitude[i] * sin(2*pi*freq*t + phase[i])."""
    dof = wp.tid()
    target_q[dof] = bias[dof] + amplitude[dof] * wp.sin(2.0 * wp.pi * freq * t + phase[dof])


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


class Arm7DofScenario(Scenario):
    id = "arm_7dof"
    supported_solvers = ("xpbd", "featherstone", "mujoco")

    # Franka arm only (no hand/fingers): first 7 revolute joints.
    ARM_DOFS_PER_WORLD = 7
    KP = 120.0
    KD = 15.0
    TAU_LIMIT = 80.0  # N*m
    TRAJ_FREQ = 0.25  # Hz

    def use_ground_plane(self) -> bool:
        return self.args.solver.name != "featherstone"

    def _on_built(self) -> None:
        self._q_hist: list[np.ndarray] = []
        self._qd_hist: list[np.ndarray] = []
        self._tau_hist: list[np.ndarray] = []
        self._contact_hist: list[int] = []

        assert self.model is not None and self.control is not None

        total_dofs = int(self.model.joint_dof_count)
        self._target_q = wp.zeros(total_dofs, dtype=wp.float32)

        # PD gains, torque limits, and per-DOF trajectory parameters.
        kp = np.zeros(total_dofs, dtype=np.float32)
        kd = np.zeros(total_dofs, dtype=np.float32)
        lim = np.zeros(total_dofs, dtype=np.float32)
        amp = np.zeros(total_dofs, dtype=np.float32)
        phase = np.zeros(total_dofs, dtype=np.float32)
        bias = np.zeros(total_dofs, dtype=np.float32)

        dofs_per_world = total_dofs // self.args.world_count
        self._dofs_per_world = dofs_per_world

        kp_value = 20.0 if self.args.solver.name == "featherstone" else self.KP
        kd_value = 3.0 if self.args.solver.name == "featherstone" else self.KD
        tau_limit = 15.0 if self.args.solver.name == "featherstone" else self.TAU_LIMIT

        for w in range(self.args.world_count):
            off = w * dofs_per_world
            for i in range(min(self.ARM_DOFS_PER_WORLD, dofs_per_world)):
                kp[off + i] = kp_value
                kd[off + i] = kd_value
                lim[off + i] = tau_limit
                # Same trajectory in every world so determinism is purely a
                # physics property, not a scheduling one.
                amp[off + i] = 0.35 * float((-1) ** i)
                phase[off + i] = float(i) * 0.4
                # Use the current joint_q as the bias so the trajectory
                # stays near the configured home pose.
                bias[off + i] = float(self.model.joint_q.numpy()[off + i])

        self._kp = wp.array(kp, dtype=wp.float32)
        self._kd = wp.array(kd, dtype=wp.float32)
        self._tau_limit = wp.array(lim, dtype=wp.float32)
        self._amp = wp.array(amp, dtype=wp.float32)
        self._phase = wp.array(phase, dtype=wp.float32)
        self._bias = wp.array(bias, dtype=wp.float32)

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        if self.args.solver.name == "featherstone":
            builder.default_joint_cfg.armature = 0.05

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.mu = 0.5

        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"
        base_height = 1.0 if self.args.solver.name == "featherstone" else 0.0
        builder.add_urdf(
            str(urdf_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, base_height), wp.quat_identity()),
            floating=False,  # arm is base-fixed
            enable_self_collisions=False,
        )
        if self.args.solver.name == "featherstone":
            builder.joint_armature[:] = [0.05] * len(builder.joint_armature)

        # Seed joint_q to the Franka home pose (arm DOFs only; fingers stay
        # at 0). These values are the ones used in example_robot_panda_hydro.
        init_q = [
            -3.68e-03,
            2.39e-02,
            3.68e-03,
            -2.368,
            -1.29e-04,
            2.392,
            0.785398,
        ]
        for i, q in enumerate(init_q[: len(builder.joint_q)]):
            builder.joint_q[i] = q

    def per_step(self) -> None:
        assert self.model is not None and self.control is not None and self.state_0 is not None

        if self.args.solver.name == "xpbd":
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Use sim-time derived from step_index so the trajectory is
        # deterministic independent of wall clock.
        t = float(self.step_index / self.args.fps)  # frame-aligned sim time

        total_dofs = int(self.model.joint_dof_count)

        wp.launch(
            _write_targets,
            dim=total_dofs,
            inputs=[t, self._amp, self._phase, self._bias, self.TRAJ_FREQ],
            outputs=[self._target_q],
        )
        wp.launch(
            _pd_control,
            dim=total_dofs,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._target_q,
                self._kp,
                self._kd,
                self._tau_limit,
            ],
            outputs=[self.control.joint_f],
        )

    def step(self) -> None:  # type: ignore[override]
        super().step()
        assert self.state_0 is not None
        if self.step_index % max(1, self.args.substeps) == 0:
            q = self.state_0.joint_q.numpy().copy()
            qd = self.state_0.joint_qd.numpy().copy()
            tau = self.control.joint_f.numpy().copy()
            self._q_hist.append(q)
            self._qd_hist.append(qd)
            self._tau_hist.append(tau)
            self._contact_hist.append(int(self.contacts.rigid_contact_count.numpy()[0]))

    def extra_snapshot(self) -> dict[str, Any]:
        assert self.model is not None and self.state_0 is not None
        total_dofs = int(self.model.joint_dof_count)
        dpw = self._dofs_per_world

        q = self.state_0.joint_q.numpy()
        per_world_q = q.reshape(self.args.world_count, dpw)
        arm_q = per_world_q[:, : self.ARM_DOFS_PER_WORLD]
        target = self._target_q.numpy().reshape(self.args.world_count, dpw)
        arm_tgt = target[:, : self.ARM_DOFS_PER_WORLD]
        err = np.abs(arm_q - arm_tgt)
        final_target_error = err.max(axis=1)

        def _stack(h: list[np.ndarray]) -> np.ndarray:
            if not h:
                return np.zeros((0, self.args.world_count, self.ARM_DOFS_PER_WORLD), dtype=np.float32)
            arr = np.stack(h, axis=0).reshape(len(h), self.args.world_count, dpw)
            return arr[:, :, : self.ARM_DOFS_PER_WORLD].astype(np.float32)

        return {
            "joint_q_history": _stack(self._q_hist),
            "joint_qd_history": _stack(self._qd_hist),
            "joint_tau_history": _stack(self._tau_hist),
            "final_target_error": final_target_error.astype(np.float32),
            "contact_count_history": np.asarray(self._contact_hist, dtype=np.int32),
            "total_joint_dofs": total_dofs,
            "dofs_per_world": dpw,
        }
