"""Sliced per-world consumed-return counter lowering probe."""

from __future__ import annotations

import numpy as np
import warp as wp

from .warp_lowering_gaps import WarpMicroScenario


@wp.kernel
def _sliced_counter_kernel(
    worlds: wp.array[wp.int32],
    lanes: wp.array[wp.int32],
    increments: wp.array[wp.int32],
    counts: wp.array2d[wp.int32],
    slots: wp.array3d[wp.int32],
):
    tid = wp.tid()
    world = worlds[tid]
    lane = lanes[tid]
    inc = increments[tid]

    world_counts = counts[world]
    start = wp.atomic_add(world_counts, lane, inc)

    for i in range(inc):
        slots[world, lane, start + i] = tid * 16 + i


class WarpSlicedCounterScenario(WarpMicroScenario):
    """Consumed-return ``atomic_add`` through a sliced per-world counter view."""

    id = "warp_counter_sliced"
    expected_current_gap = "sliced counter target with dynamic prefix/index"

    def _build_workload(self) -> None:
        n = 96
        world_count = max(2, self.args.world_count)
        lane_count = 3

        tids = np.arange(n, dtype=np.int32)
        worlds = (tids * 5 + 1) % world_count
        lanes = (tids * 7 + 2) % lane_count
        increments = (tids % 2) + 1

        per_lane = np.zeros((world_count, lane_count), dtype=np.int32)
        for world, lane, inc in zip(worlds, lanes, increments, strict=True):
            per_lane[world, lane] += inc
        capacity = int(per_lane.max())

        self.worlds = wp.array(worlds, dtype=wp.int32, device=self.device)
        self.lanes = wp.array(lanes, dtype=wp.int32, device=self.device)
        self.increments = wp.array(increments, dtype=wp.int32, device=self.device)
        self.counts = wp.zeros((world_count, lane_count), dtype=wp.int32, device=self.device)
        self.slots = wp.full((world_count, lane_count, capacity), -1, dtype=wp.int32, device=self.device)

    def _run_workload(self) -> None:
        self.counts.zero_()
        self.slots.fill_(-1)
        wp.launch(
            _sliced_counter_kernel,
            dim=self.worlds.shape[0],
            inputs=[self.worlds, self.lanes, self.increments],
            outputs=[self.counts, self.slots],
            device=self.device,
        )

    def _core_arrays(self) -> dict[str, np.ndarray]:
        return {
            "counts": self.counts.numpy().copy(),
            "slots": self.slots.numpy().copy(),
        }
