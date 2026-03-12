# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: per-world mesh randomization via GPU-resident caching.
#
# One scene with both kinds of randomization running simultaneously:
#
#   - Object A: single geom, 2 mesh variants (small / large)      [geom-level]
#   - Object B: single geom, 2 mesh variants (medium / tiny)      [geom-level]
#   - Object C: 4 geom slots (convex decomposition), 3 variants   [body-level]
#       Variant X (4 pieces) ← default, most complex
#       Variant Y (3 pieces) — slot 3 disabled
#       Variant Z (2 pieces) — slots 2-3 disabled
#   - Arm D: 2-link arm (base→link1→link2, revolute joints)       [articulated]
#       link1: 2 mesh variants, link2: 2 mesh variants
#       Verifies set_const recomputes body_subtreemass up the chain.
#
# MeshRandomizer sets model.shape_active_variant per world and calls
# notify_model_changed(MESH_VARIANT_PROPERTIES).  The solver handles
# all cache lookups, geom-field updates, body inertia recomputation,
# and derived-quantity sync (subtreemass, invweight0).
#
# Requires: mujoco_warp with PR #1191 (2D geom_dataid).
#
# Run:  uv run --extra dev python newton/tests/prototype_mesh_randomization.py

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import newton
from newton._src.solvers.flags import SolverNotifyFlags

NWORLD = 64

# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

BOX_INDICES = np.array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
], dtype=np.int32)


def _box_verts(hx: float, hy: float, hz: float, offset=(0., 0., 0.)):
    ox, oy, oz = offset
    return np.array([
        [ox - hx, oy - hy, oz - hz], [ox + hx, oy - hy, oz - hz],
        [ox + hx, oy + hy, oz - hz], [ox - hx, oy + hy, oz - hz],
        [ox - hx, oy - hy, oz + hz], [ox + hx, oy - hy, oz + hz],
        [ox + hx, oy + hy, oz + hz], [ox - hx, oy + hy, oz + hz],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# MeshRandomizer — thin wrapper that sets shape_active_variant per group
# ---------------------------------------------------------------------------


class _VariantGroup:
    """One independent randomization group (one body, one or more shapes)."""

    def __init__(self):
        self.body_name: str = ""
        self.n_variants: int = 0
        self.shape_indices: list[int] = []  # template shape indices (within one world)


class MeshRandomizer:
    """Per-world mesh randomizer that auto-discovers variant groups from the solver.

    Groups shapes by body: each body with mesh variants forms one independent
    randomization group.

    Usage::

        randomizer = MeshRandomizer(solver)
        indices = randomizer.reset(solver, nworld, rng)
    """

    def __init__(self, solver):
        self.groups: list[_VariantGroup] = []

        model = solver.model
        bodies_per_world = model.body_count // model.world_count
        shape_types = model.shape_type.numpy()
        first_env = solver._first_env_shape_base
        shapes_per_world = solver._shapes_per_world

        for body_id in range(bodies_per_world):
            shapes = model.body_shapes.get(body_id, [])
            mesh_shapes = [
                s for s in shapes
                if shape_types[s] in (7, 10) and model.shape_mesh_variants[s]
            ]
            if not mesh_shapes:
                continue

            all_mesh = [s for s in shapes if shape_types[s] in (7, 10)]
            n_variants = 1 + max(len(model.shape_mesh_variants[s]) for s in mesh_shapes)

            group = _VariantGroup()
            group.body_name = model.body_label[body_id]
            group.n_variants = n_variants
            group.shape_indices = all_mesh
            self.groups.append(group)

        self._first_env = first_env
        self._shapes_per_world = shapes_per_world

    def reset(
        self, solver, nworld: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Randomize all groups.  Returns per-group variant index arrays."""
        sav = solver.model.shape_active_variant.numpy()
        all_indices = []

        for group in self.groups:
            local_idx = rng.integers(0, group.n_variants, size=nworld).astype(np.int32)
            all_indices.append(local_idx)
            for w in range(nworld):
                for s in group.shape_indices:
                    sav[self._first_env + (s - self._first_env) + w * self._shapes_per_world] = local_idx[w]

        solver.model.shape_active_variant.assign(sav)
        solver.notify_model_changed(SolverNotifyFlags.MESH_VARIANT_PROPERTIES)
        return all_indices


# ===========================================================================
# Demo
# ===========================================================================

def main():
    # =====================================================================
    # Step 1: Build the model
    # =====================================================================

    world_builder = newton.ModelBuilder()

    body_a = world_builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), label="obj_a")
    world_builder.add_shape_mesh(
        body=body_a,
        mesh=newton.Mesh(_box_verts(0.1, 0.1, 0.1), BOX_INDICES),
        label="geom_a",
        mesh_variants=[newton.Mesh(_box_verts(0.2, 0.2, 0.2), BOX_INDICES)],
    )

    body_b = world_builder.add_body(
        xform=wp.transform((0.0, 1.0, 0.5), wp.quat_identity()), label="obj_b")
    world_builder.add_shape_mesh(
        body=body_b,
        mesh=newton.Mesh(_box_verts(0.15, 0.15, 0.15), BOX_INDICES),
        label="geom_b",
        mesh_variants=[newton.Mesh(_box_verts(0.05, 0.05, 0.05), BOX_INDICES)],
    )

    hull_x = [_box_verts(0.08, 0.08, 0.04, offset=(i * 0.16, 0, 0)) for i in range(4)]
    hull_y = [
        _box_verts(0.12, 0.06, 0.06),
        _box_verts(0.12, 0.06, 0.06, offset=(0.24, 0, 0)),
        _box_verts(0.06, 0.12, 0.06, offset=(0.12, 0.12, 0)),
    ]
    hull_z = [
        _box_verts(0.15, 0.15, 0.08),
        _box_verts(0.15, 0.15, 0.08, offset=(0.3, 0, 0)),
    ]

    body_c = world_builder.add_body(
        xform=wp.transform((0.0, 2.0, 0.5), wp.quat_identity()), label="obj_c")
    world_builder.add_shape_mesh(
        body=body_c, mesh=newton.Mesh(hull_x[0], BOX_INDICES), label="hull_0",
        mesh_variants=[newton.Mesh(hull_y[0], BOX_INDICES),
                       newton.Mesh(hull_z[0], BOX_INDICES)])
    world_builder.add_shape_mesh(
        body=body_c, mesh=newton.Mesh(hull_x[1], BOX_INDICES), label="hull_1",
        mesh_variants=[newton.Mesh(hull_y[1], BOX_INDICES),
                       newton.Mesh(hull_z[1], BOX_INDICES)])
    world_builder.add_shape_mesh(
        body=body_c, mesh=newton.Mesh(hull_x[2], BOX_INDICES), label="hull_2",
        mesh_variants=[newton.Mesh(hull_y[2], BOX_INDICES)])
    world_builder.add_shape_mesh(
        body=body_c, mesh=newton.Mesh(hull_x[3], BOX_INDICES), label="hull_3",
        mesh_variants=[])

    # D — 2-link articulated arm: base (fixed to world) → link1 → link2
    arm_base = world_builder.add_link(
        xform=wp.transform((0.0, 3.0, 0.5), wp.quat_identity()), label="arm_base")
    world_builder.add_shape_mesh(
        body=arm_base,
        mesh=newton.Mesh(_box_verts(0.05, 0.05, 0.05), BOX_INDICES),
        label="arm_base_geom",
    )

    arm_link1 = world_builder.add_link(
        xform=wp.transform((0.0, 3.0, 0.8), wp.quat_identity()), label="arm_link1")
    world_builder.add_shape_mesh(
        body=arm_link1,
        mesh=newton.Mesh(_box_verts(0.04, 0.04, 0.15), BOX_INDICES),
        label="link1_geom",
        mesh_variants=[newton.Mesh(_box_verts(0.06, 0.06, 0.15), BOX_INDICES)],
    )

    arm_link2 = world_builder.add_link(
        xform=wp.transform((0.0, 3.0, 1.1), wp.quat_identity()), label="arm_link2")
    world_builder.add_shape_mesh(
        body=arm_link2,
        mesh=newton.Mesh(_box_verts(0.03, 0.03, 0.12), BOX_INDICES),
        label="link2_geom",
        mesh_variants=[newton.Mesh(_box_verts(0.05, 0.05, 0.12), BOX_INDICES)],
    )

    j0 = world_builder.add_joint_fixed(parent=-1, child=arm_base)
    j1 = world_builder.add_joint_revolute(
        parent=arm_base, child=arm_link1, axis=newton.Axis.Y)
    j2 = world_builder.add_joint_revolute(
        parent=arm_link1, child=arm_link2, axis=newton.Axis.Y)
    world_builder.add_articulation([j0, j1, j2], label="arm")

    main_builder = newton.ModelBuilder()
    main_builder.add_ground_plane()
    main_builder.replicate(world_builder, world_count=NWORLD, spacing=(2.0, 0.0, 0.0))

    model = main_builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model, iterations=1)
    mj_model = solver.mj_model
    nworld = solver.mjw_data.nworld

    print(f"\n{nworld} worlds, nbody={mj_model.nbody}, ngeom={mj_model.ngeom}, nmesh={mj_model.nmesh}")
    print(f"Mesh variant cache built: {solver._has_mesh_variants}")

    # =====================================================================
    # Step 2: Initialize the randomizer
    # =====================================================================

    randomizer = MeshRandomizer(solver)
    print(f"Auto-discovered {len(randomizer.groups)} randomization groups.")

    # =====================================================================
    # Step 3: Randomize (one call per episode reset)
    # =====================================================================

    rng = np.random.default_rng(42)
    indices = randomizer.reset(solver, nworld, rng)

    # =====================================================================
    # Step 4: Verify
    # =====================================================================

    # Verify: all worlds with the same variant index must have identical mass.
    mass = solver.mjw_model.body_mass.numpy()
    for gi, group in enumerate(randomizer.groups):
        mj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, group.body_name)
        idx = indices[gi]
        for vi in range(group.n_variants):
            world_masses = mass[idx == vi, mj_body_id]
            assert len(set(world_masses)) <= 1, (
                f"{group.body_name} variant {vi}: inconsistent masses {set(world_masses)}"
            )

    # Verify: subtreemass accumulates correctly for the articulated arm.
    subtreemass = solver.mjw_model.body_subtreemass.numpy()
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
    l1_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link1")
    l2_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link2")
    for w in range(nworld):
        expected = mass[w, base_id] + mass[w, l1_id] + mass[w, l2_id]
        assert abs(subtreemass[w, base_id] - expected) < 0.01, (
            f"world {w}: subtreemass {subtreemass[w, base_id]:.2f} != {expected:.2f}"
        )

    # Verify: simulation step produces finite state.
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))

    print("All checks passed.")


if __name__ == "__main__":
    main()
