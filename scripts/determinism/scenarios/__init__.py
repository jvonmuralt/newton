"""Scenario registry for the determinism harness.

Add new scenarios by subclassing :class:`scripts.determinism.harness.Scenario`
and registering the class here.
"""

from __future__ import annotations

from .arm_7dof import Arm7DofScenario
from .box_stack import BoxStackScenario
from .diffsim_ball import DiffsimBallScenario
from .diffsim_cloth_com import DiffsimClothComScenario
from .diffsim_spring_cage import DiffsimSpringCageScenario
from .domino_chain import DominoChainScenario
from .falling_cube import FallingCubeScenario
from .humanoid import HumanoidScenario
from .warp_argmax_exchange import WarpArgmaxExchangeScenario
from .warp_counter_indexed import WarpIndexedCounterScenario
from .warp_counter_sliced import WarpSlicedCounterScenario
from .warp_counter_static_index import WarpStaticCounterScenario
from .warp_custom_adjoint import WarpCustomAdjointScenario

SCENARIOS = {
    FallingCubeScenario.id: FallingCubeScenario,
    BoxStackScenario.id: BoxStackScenario,
    DiffsimBallScenario.id: DiffsimBallScenario,
    DiffsimClothComScenario.id: DiffsimClothComScenario,
    DiffsimSpringCageScenario.id: DiffsimSpringCageScenario,
    DominoChainScenario.id: DominoChainScenario,
    Arm7DofScenario.id: Arm7DofScenario,
    HumanoidScenario.id: HumanoidScenario,
    WarpIndexedCounterScenario.id: WarpIndexedCounterScenario,
    WarpStaticCounterScenario.id: WarpStaticCounterScenario,
    WarpSlicedCounterScenario.id: WarpSlicedCounterScenario,
    WarpCustomAdjointScenario.id: WarpCustomAdjointScenario,
    WarpArgmaxExchangeScenario.id: WarpArgmaxExchangeScenario,
}

__all__ = [
    "SCENARIOS",
    "Arm7DofScenario",
    "BoxStackScenario",
    "DiffsimBallScenario",
    "DiffsimClothComScenario",
    "DiffsimSpringCageScenario",
    "DominoChainScenario",
    "FallingCubeScenario",
    "HumanoidScenario",
    "WarpArgmaxExchangeScenario",
    "WarpCustomAdjointScenario",
    "WarpIndexedCounterScenario",
    "WarpSlicedCounterScenario",
    "WarpStaticCounterScenario",
]
