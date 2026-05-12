"""Consumed-return max plus exchange winner lowering probe."""

from __future__ import annotations

import numpy as np
import warp as wp

from .warp_lowering_gaps import WarpMicroScenario


@wp.kernel
def _argmax_exchange_kernel(
    scores: wp.array[wp.int32],
    best_score: wp.array[wp.int32],
    best_tid: wp.array[wp.int32],
):
    tid = wp.tid()
    score = scores[tid]
    previous = wp.atomic_max(best_score, 0, score)
    if score > previous:
        wp.atomic_exch(best_tid, 0, tid)


class WarpArgmaxExchangeScenario(WarpMicroScenario):
    """Consumed-return ``atomic_max`` followed by an ``atomic_exch`` winner."""

    id = "warp_argmax_exchange"
    expected_current_gap = "return-consuming atomic_max plus exchange winner"

    def _build_workload(self) -> None:
        scores = np.array([2, 7, 3, 11, 5, 13, 13, 4, 9, 1, 12, 8], dtype=np.int32)
        self.scores = wp.array(scores, dtype=wp.int32, device=self.device)
        self.best_score = wp.full(1, -2147483648, dtype=wp.int32, device=self.device)
        self.best_tid = wp.full(1, -1, dtype=wp.int32, device=self.device)

    def _run_workload(self) -> None:
        self.best_score.fill_(-2147483648)
        self.best_tid.fill_(-1)
        wp.launch(
            _argmax_exchange_kernel,
            dim=self.scores.shape[0],
            inputs=[self.scores],
            outputs=[self.best_score, self.best_tid],
            device=self.device,
        )

    def _core_arrays(self) -> dict[str, np.ndarray]:
        return {
            "best_score": self.best_score.numpy().copy(),
            "best_tid": self.best_tid.numpy().copy(),
        }
