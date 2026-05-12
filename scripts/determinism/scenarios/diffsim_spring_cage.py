"""Differentiable spring rest-length target gradient."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from ..harness import Scenario, ScenarioSnapshot


@wp.kernel
def _spring_cage_loss(
    particle_q: wp.array[wp.vec3],
    target: wp.vec3,
    loss: wp.array[wp.float32],
):
    loss[0] = wp.length_sq(particle_q[0] - target)


class DiffsimSpringCageScenario(Scenario):
    """Spring rest-length gradient inspired by ``example_diffsim_spring_cage``."""

    id = "diffsim_spring_cage"
    supported_solvers = ("semi_implicit",)

    def build_subworld(self, builder: newton.ModelBuilder) -> None:
        del builder

    def build(self, viewer: Any) -> None:
        self.horizon_steps = max(1, self.args.num_steps)
        self.horizon_substeps = max(1, self.args.substeps)
        self.sim_dt = 1.0 / self.args.fps / self.horizon_substeps
        self.target = wp.vec3(0.375, 0.125, 0.25)
        self.loss_history: list[float] = []

        scene = newton.ModelBuilder()
        scene.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        cage_points = (
            (0.2, -0.7, 0.8),
            (1.1, 0.0, 0.2),
            (-1.2, 0.1, 0.1),
            (0.4, 0.6, 0.4),
            (-0.2, 0.7, -0.9),
            (0.1, -0.8, -0.8),
            (-0.8, -0.9, 0.2),
            (-0.1, 1.0, 0.4),
        )
        for point in cage_points:
            scene.add_particle(point, (0.0, 0.0, 0.0), 0.0)

        for particle in range(1, scene.particle_count):
            scene.add_spring(0, particle, 100.0, 10.0, 0.0)

        self.model = scene.finalize(requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        state_count = self.horizon_steps * self.horizon_substeps
        self.states = [self.model.state() for _ in range(state_count + 1)]
        self.control = self.model.control()
        self.contacts = None
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self._ran_backward = False

        self.viewer = viewer
        viewer.set_model(self.model)

    def _forward(self) -> wp.array[wp.float32]:
        assert self.solver is not None and self.control is not None
        self.loss.zero_()

        for step in range(self.horizon_steps):
            for substep in range(self.horizon_substeps):
                idx = step * self.horizon_substeps + substep
                self.states[idx].clear_forces()
                self.solver.step(
                    self.states[idx],
                    self.states[idx + 1],
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )

        wp.launch(
            _spring_cage_loss,
            dim=1,
            inputs=[self.states[-1].particle_q, self.target],
            outputs=[self.loss],
        )
        return self.loss

    def _forward_backward(self) -> None:
        for state in self.states:
            state.particle_q.grad.zero_()
            state.particle_qd.grad.zero_()
        self.model.spring_rest_length.grad.zero_()
        self.loss.grad.zero_()

        tape = wp.Tape()
        with tape:
            loss = self._forward()
        tape.backward(loss)
        self.loss_history.append(float(self.loss.numpy()[0]))
        self._ran_backward = True

    def step(self) -> None:
        if not self._ran_backward:
            self._forward_backward()
        self.step_index += 1

    def render(self) -> None:
        state_index = min(self.step_index * self.horizon_substeps, len(self.states) - 1)
        state = self.states[state_index]
        self.viewer.begin_frame(self.step_index * (1.0 / self.args.fps))
        self.viewer.log_state(state)
        self.viewer.end_frame()

    def snapshot(self) -> ScenarioSnapshot:
        if not self._ran_backward:
            self._forward_backward()

        core = {
            "final_particle_q": self.states[-1].particle_q.numpy().copy(),
            "final_particle_qd": self.states[-1].particle_qd.numpy().copy(),
            "spring_rest_length_grad": self.model.spring_rest_length.grad.numpy().copy(),
            "loss": self.loss.numpy().copy(),
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
            "graph_capture_enabled": False,
            "diffsim": True,
        }
        extras = {
            "loss_history": np.asarray(self.loss_history, dtype=np.float32),
            "target": np.asarray(tuple(self.target), dtype=np.float32),
        }
        return ScenarioSnapshot(core=core, extras=extras, meta=meta)
