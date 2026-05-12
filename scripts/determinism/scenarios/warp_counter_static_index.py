"""Nonzero static consumed-return counter index lowering probe."""

from __future__ import annotations

import numpy as np
import warp as wp

from .warp_lowering_gaps import WarpMicroScenario


@wp.kernel
def _static_counter_kernel(
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
):
    tid = wp.tid()
    start = wp.atomic_add(counts, wp.static(1), 1)
    slots[start] = tid


class WarpStaticCounterScenario(WarpMicroScenario):
    """Consumed-return ``atomic_add`` with a nonzero ``wp.static`` index."""

    id = "warp_counter_static_index"
    expected_current_gap = "nonzero static consumed-return counter index"

    def _build_workload(self) -> None:
        self.n = 32
        self.counts = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.slots = wp.full(self.n, -1, dtype=wp.int32, device=self.device)

    def _run_workload(self) -> None:
        self.counts.zero_()
        self.slots.fill_(-1)
        wp.launch(
            _static_counter_kernel,
            dim=self.n,
            outputs=[self.counts, self.slots],
            device=self.device,
        )

    def _core_arrays(self) -> dict[str, np.ndarray]:
        return {
            "counts": self.counts.numpy().copy(),
            "slots": self.slots.numpy().copy(),
        }
