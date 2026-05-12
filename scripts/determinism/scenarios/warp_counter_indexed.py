"""Dynamic consumed-return counter index lowering probe."""

from __future__ import annotations

import numpy as np
import warp as wp

from .warp_lowering_gaps import WarpMicroScenario


@wp.kernel
def _indexed_counter_kernel(
    bins: wp.array[wp.int32],
    increments: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slots: wp.array2d[wp.int32],
):
    tid = wp.tid()
    bin_id = bins[tid]
    inc = increments[tid]
    start = wp.atomic_add(counts, bin_id, inc)

    for i in range(inc):
        slots[bin_id, start + i] = tid * 16 + i


class WarpIndexedCounterScenario(WarpMicroScenario):
    """Consumed-return ``atomic_add`` with a dynamic counter index."""

    id = "warp_counter_indexed"
    expected_current_gap = "dynamic consumed-return counter index"

    def _build_workload(self) -> None:
        n = 64
        bin_count = 4
        bins = (np.arange(n, dtype=np.int32) * 3 + 1) % bin_count
        increments = (np.arange(n, dtype=np.int32) % 3) + 1
        capacity = int(np.bincount(bins, weights=increments, minlength=bin_count).max())

        self.bins = wp.array(bins, dtype=wp.int32, device=self.device)
        self.increments = wp.array(increments, dtype=wp.int32, device=self.device)
        self.counts = wp.zeros(bin_count, dtype=wp.int32, device=self.device)
        self.slots = wp.full((bin_count, capacity), -1, dtype=wp.int32, device=self.device)

    def _run_workload(self) -> None:
        self.counts.zero_()
        self.slots.fill_(-1)
        wp.launch(
            _indexed_counter_kernel,
            dim=self.bins.shape[0],
            inputs=[self.bins, self.increments],
            outputs=[self.counts, self.slots],
            device=self.device,
        )

    def _core_arrays(self) -> dict[str, np.ndarray]:
        return {
            "counts": self.counts.numpy().copy(),
            "slots": self.slots.numpy().copy(),
        }
