"""Custom ``wp.adjoint[array]`` deterministic target mapping probe."""

from __future__ import annotations

import numpy as np
import warp as wp

from .warp_lowering_gaps import WarpMicroScenario


@wp.func
def _custom_adjoint_square(i: int, x: wp.array[float], y: wp.array[float]):
    y[i] = x[i] * x[i]


@wp.func_grad(_custom_adjoint_square)
def _adj_custom_adjoint_square(i: int, x: wp.array[float], y: wp.array[float]):
    wp.adjoint[x][i] += 2.0 * x[i] * wp.adjoint[y][i]


@wp.kernel
def _custom_adjoint_kernel(x: wp.array[float], y: wp.array[float]):
    tid = wp.tid()
    _custom_adjoint_square(tid, x, y)


class WarpCustomAdjointScenario(WarpMicroScenario):
    """Custom ``@wp.func_grad`` that accumulates into ``wp.adjoint[array]``."""

    id = "warp_custom_adjoint"
    expected_current_gap = "deterministic target mapping for wp.adjoint[array]"
    graph_capture_enabled = False

    def _build_workload(self) -> None:
        values = np.linspace(0.5, 2.0, 16, dtype=np.float32)
        self.x = wp.array(values, dtype=wp.float32, requires_grad=True, device=self.device)
        self.y = wp.zeros_like(self.x, requires_grad=True)
        self.y_grad = wp.ones_like(self.y)

    def _run_workload(self) -> None:
        self.x.grad.zero_()
        self.y.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(
                _custom_adjoint_kernel,
                dim=self.x.shape[0],
                inputs=[self.x],
                outputs=[self.y],
                device=self.device,
            )
        tape.backward(grads={self.y: self.y_grad})

    def _core_arrays(self) -> dict[str, np.ndarray]:
        return {
            "y": self.y.numpy().copy(),
            "x_grad": self.x.grad.numpy().copy(),
        }
