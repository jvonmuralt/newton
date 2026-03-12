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
# MeshRandomizer auto-discovers all variant groups from the solver and
# manages randomization via a single reset() call.
#
# Requires: mujoco_warp with PR #1191 (2D geom_dataid).
#
# Run:  uv run --extra dev python newton/tests/prototype_mesh_randomization.py

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import newton

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
# Warp scatter kernels — one for geom fields, one for body fields
# ---------------------------------------------------------------------------


@wp.kernel
def _scatter_geom(
    idx: wp.array(dtype=int),
    geom_ids: wp.array(dtype=int),
    dataid_src: wp.array2d(dtype=int),    dataid_dst: wp.array2d(dtype=int),
    size_src: wp.array2d(dtype=wp.vec3),  size_dst: wp.array2d(dtype=wp.vec3),
    rbound_src: wp.array2d(dtype=float),  rbound_dst: wp.array2d(dtype=float),
    pos_src: wp.array2d(dtype=wp.vec3),   pos_dst: wp.array2d(dtype=wp.vec3),
):
    w, i = wp.tid()
    gid = geom_ids[i]
    vi = idx[w]
    dataid_dst[w, gid] = dataid_src[vi, i]
    size_dst[w, gid] = size_src[vi, i]
    rbound_dst[w, gid] = rbound_src[vi, i]
    pos_dst[w, gid] = pos_src[vi, i]


@wp.kernel
def _scatter_body(
    idx: wp.array(dtype=int),
    body_ids: wp.array(dtype=int),
    mass_src: wp.array2d(dtype=float),    mass_dst: wp.array2d(dtype=float),
    inertia_src: wp.array2d(dtype=wp.vec3), inertia_dst: wp.array2d(dtype=wp.vec3),
    ipos_src: wp.array2d(dtype=wp.vec3),    ipos_dst: wp.array2d(dtype=wp.vec3),
    iquat_src: wp.array2d(dtype=wp.quat),   iquat_dst: wp.array2d(dtype=wp.quat),
):
    w, i = wp.tid()
    bid = body_ids[i]
    vi = idx[w]
    mass_dst[w, bid] = mass_src[vi, i]
    inertia_dst[w, bid] = inertia_src[vi, i]
    ipos_dst[w, bid] = ipos_src[vi, i]
    iquat_dst[w, bid] = iquat_src[vi, i]

# ---------------------------------------------------------------------------
# MeshRandomizer — auto-discovers variant groups from solver, single reset()
# ---------------------------------------------------------------------------


class _VariantGroup:
    """One independent randomization group (one body, one or more geom slots)."""

    def __init__(self):
        self.body_name: str = ""
        self.n_variants: int = 0
        self.geom_ids: wp.array = None             # (n_geoms,) int, on GPU
        self.body_ids: wp.array = None             # (n_bodies,) int, on GPU
        self.gpu_cache: dict[str, wp.array] = {}   # field → (n_variants, n_owned)


class MeshRandomizer:
    """Per-world mesh randomizer that auto-discovers variant groups from the solver.

    Groups shapes by body: each body with mesh variants forms one independent
    randomization group.  Geom-level (1 shape on a body) and body-level
    (multiple shapes on a body, e.g. convex decomposition) are handled
    uniformly.

    Usage::

        randomizer = MeshRandomizer(solver)
        indices = randomizer.reset(solver.mjw_model, solver.mjw_data, nworld, rng)
    """

    def __init__(self, solver, device: str = "cuda:0"):
        self.device = device
        self.groups: list[_VariantGroup] = []

        newton_model = solver.model
        bodies_per_world = newton_model.body_count // newton_model.world_count
        shape_types = newton_model.shape_type.numpy()

        for body_id in range(bodies_per_world):
            mesh_shapes = self._find_mesh_shapes(newton_model, body_id, shape_types)
            if not mesh_shapes:
                continue

            variant_counts = [len(newton_model.shape_mesh_variants[s]) for s in mesh_shapes]
            if max(variant_counts) == 0:
                continue

            n_variants = 1 + max(variant_counts)
            variants = self._build_variant_dicts(newton_model, mesh_shapes, n_variants)

            group = _VariantGroup()
            group.body_name = newton_model.body_label[body_id]
            group.n_variants = n_variants
            geom_name_to_id = self._resolve_mujoco_ids(
                group, variants, solver.mj_model, device,
            )
            self._build_dataid_cache(group, variants, geom_name_to_id, solver.mj_model, device)
            self._compute_and_cache(group, newton_model, mesh_shapes, device)
            self.groups.append(group)

    # -- Discovery --

    @staticmethod
    def _find_mesh_shapes(newton_model, body_id: int, shape_types: np.ndarray) -> list[int]:
        """Return mesh-type shape indices for a given body."""
        shapes = newton_model.body_shapes.get(body_id, [])
        return [s for s in shapes if shape_types[s] == 7]  # GeoType.MESH

    @staticmethod
    def _build_variant_dicts(
        newton_model, mesh_shapes: list[int], n_variants: int
    ) -> list[dict[str, str | None]]:
        """Build per-variant dicts: {geom_name: mesh_name | None}.

        Variant 0 = original meshes.
        Variant i>0 = mesh_variants[i-1] per shape, or None if that shape
        has fewer variants (disables the geom slot).
        """
        variants = []
        for vi in range(n_variants):
            var_dict = {}
            for s in mesh_shapes:
                geom_name = f"{newton_model.shape_label[s]}_{s}"
                if vi == 0:
                    var_dict[geom_name] = geom_name
                else:
                    mvs = newton_model.shape_mesh_variants[s]
                    if vi - 1 < len(mvs):
                        var_dict[geom_name] = f"{geom_name}_variant_{vi - 1}"
                    else:
                        var_dict[geom_name] = None
            variants.append(var_dict)
        return variants

    # -- MuJoCo id resolution --

    @staticmethod
    def _resolve_mujoco_ids(
        group: _VariantGroup, variants: list[dict], mj_model, device: str
    ) -> dict[str, int]:
        """Map geom/body names to MuJoCo integer ids.  Returns name→id dict."""
        geom_name_to_id: dict[str, int] = {}
        body_id_set: set[int] = set()
        for gn in variants[0]:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gn)
            geom_name_to_id[gn] = gid
            body_id_set.add(int(mj_model.geom_bodyid[gid]))

        group.geom_ids = wp.array(
            np.array(list(geom_name_to_id.values()), dtype=np.int32),
            dtype=int, device=device,
        )
        group.body_ids = wp.array(
            np.array(sorted(body_id_set), dtype=np.int32),
            dtype=int, device=device,
        )
        return geom_name_to_id

    @staticmethod
    def _build_dataid_cache(
        group: _VariantGroup, variants: list[dict],
        geom_name_to_id: dict[str, int], mj_model, device: str,
    ):
        """Build compact (n_variants, n_geoms) dataid cache on GPU."""
        n_geoms = len(geom_name_to_id)
        dataid = np.zeros((group.n_variants, n_geoms), dtype=np.int32)
        for vi, var in enumerate(variants):
            for gi, gname in enumerate(geom_name_to_id):
                mname = var[gname]
                dataid[vi, gi] = -1 if mname is None else mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_MESH, mname,
                )
        group.gpu_cache["geom_dataid"] = wp.array(dataid, dtype=int, device=device)

    # -- Compilation & caching --

    @staticmethod
    def _compute_and_cache(
        group: _VariantGroup,
        newton_model,
        mesh_shapes: list[int],
        device: str,
    ):
        """Compute physics properties from mesh vertices, store directly in GPU cache."""
        from newton._src.geometry.inertia import compute_inertia_mesh
        from scipy.spatial.transform import Rotation

        n_geoms = len(mesh_shapes)
        n_variants = group.n_variants
        density = 1000.0  # MuJoCo default
        all_scales = newton_model.shape_scale.numpy()

        # Pre-allocate per-variant arrays: (n_variants, n_entities, ...)
        all_geom_sizes = np.zeros((n_variants, n_geoms, 3), dtype=np.float32)
        all_geom_rbounds = np.zeros((n_variants, n_geoms), dtype=np.float32)
        all_geom_pos = np.zeros((n_variants, n_geoms, 3), dtype=np.float32)
        all_body_mass = np.zeros((n_variants, 1), dtype=np.float32)
        all_body_inertia = np.zeros((n_variants, 1, 3), dtype=np.float32)
        all_body_ipos = np.zeros((n_variants, 1, 3), dtype=np.float32)
        all_body_iquat = np.zeros((n_variants, 1, 4), dtype=np.float32)

        for vi in range(n_variants):
            geom_masses, geom_coms, geom_inertias = [], [], []

            for gi, s in enumerate(mesh_shapes):
                mvs = newton_model.shape_mesh_variants[s]
                scale = all_scales[s].astype(np.float32)

                if vi == 0:
                    mesh = newton_model.shape_source[s]
                elif vi - 1 < len(mvs):
                    mesh = mvs[vi - 1]
                else:
                    geom_masses.append(0.0)
                    geom_coms.append(np.zeros(3))
                    geom_inertias.append(np.zeros((3, 3)))
                    continue

                verts = np.array(mesh.vertices, dtype=np.float32) * scale
                indices = np.array(mesh.indices, dtype=np.int32)

                mass, com, I_mat, _ = compute_inertia_mesh(density, verts, indices)
                I_mat = np.array(I_mat).reshape(3, 3)

                vmin, vmax = verts.min(axis=0), verts.max(axis=0)
                all_geom_sizes[vi, gi] = (vmax - vmin) / 2.0
                center = (vmin + vmax) / 2.0
                all_geom_rbounds[vi, gi] = np.max(np.linalg.norm(verts - center, axis=1))

                geom_masses.append(float(mass))
                geom_coms.append(np.array(com))
                geom_inertias.append(I_mat)

            # Combine geoms → body properties
            total_mass = sum(geom_masses)
            if total_mass > 0:
                body_com = sum(m * c for m, c in zip(geom_masses, geom_coms)) / total_mass
            else:
                body_com = np.zeros(3)

            I_combined = np.zeros((3, 3))
            for m, c, I in zip(geom_masses, geom_coms, geom_inertias):
                if m > 0:
                    r = c - body_com
                    I_combined += I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

            eigenvalues, eigenvectors = np.linalg.eigh(I_combined)
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 0] *= -1
            q = Rotation.from_matrix(eigenvectors).as_quat()  # [x,y,z,w]

            all_body_mass[vi, 0] = total_mass
            all_body_inertia[vi, 0] = eigenvalues
            all_body_ipos[vi, 0] = body_com
            all_body_iquat[vi, 0] = [q[3], q[0], q[1], q[2]]  # wxyz

        # Move to GPU (update, not replace — geom_dataid is already in gpu_cache)
        group.gpu_cache.update({
            "geom_size": wp.array(all_geom_sizes, dtype=wp.vec3, device=device),
            "geom_rbound": wp.array(all_geom_rbounds, dtype=float, device=device),
            "geom_pos": wp.array(all_geom_pos, dtype=wp.vec3, device=device),
            "body_mass": wp.array(all_body_mass, dtype=float, device=device),
            "body_inertia": wp.array(all_body_inertia, dtype=wp.vec3, device=device),
            "body_ipos": wp.array(all_body_ipos, dtype=wp.vec3, device=device),
            "body_iquat": wp.array(all_body_iquat, dtype=wp.quat, device=device),
        })

    # -- Reset --

    def reset(
        self, mjw_model, mjw_data, nworld: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Randomize all groups independently.  Returns per-group index arrays."""
        all_indices = []
        for group in self.groups:
            local_idx = rng.integers(0, group.n_variants, size=nworld).astype(np.int32)
            variant_gpu = wp.array(local_idx, dtype=int, device=self.device)
            self._scatter_fields(group, mjw_model, variant_gpu, nworld)
            all_indices.append(local_idx)

        # Recompute tree-dependent fields (body_subtreemass, body_invweight0).
        mujoco_warp.set_const_fixed(mjw_model, mjw_data)
        mujoco_warp.set_const_0(mjw_model, mjw_data)

        return all_indices

    def _scatter_fields(
        self, group: _VariantGroup, mjw_model, variant_gpu: wp.array, nworld: int
    ):
        """Scatter all cached fields to model in 2 kernel launches."""
        c = group.gpu_cache
        wp.launch(
            _scatter_geom, dim=(nworld, group.geom_ids.shape[0]),
            inputs=[
                variant_gpu, group.geom_ids,
                c["geom_dataid"], mjw_model.geom_dataid,
                c["geom_size"],   mjw_model.geom_size,
                c["geom_rbound"], mjw_model.geom_rbound,
                c["geom_pos"],    mjw_model.geom_pos,
            ],
            device=self.device,
        )
        wp.launch(
            _scatter_body, dim=(nworld, group.body_ids.shape[0]),
            inputs=[
                variant_gpu, group.body_ids,
                c["body_mass"],    mjw_model.body_mass,
                c["body_inertia"], mjw_model.body_inertia,
                c["body_ipos"],    mjw_model.body_ipos,
                c["body_iquat"],   mjw_model.body_iquat,
            ],
            device=self.device,
        )


# ===========================================================================
# Demo
# ===========================================================================

def main():
    # =====================================================================
    # Step 1: Build the model
    # =====================================================================
    # Three objects with mesh variants registered via add_shape_mesh():
    #   A, B — single geom each, 2 mesh sizes           (geom-level)
    #   C    — 4 geom slots (convex decomp), 3 variants  (body-level)

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
    # Uses add_link() + add_articulation() for a proper kinematic chain.
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

    # =====================================================================
    # Step 2: Initialize the randomizer
    # =====================================================================
    # Auto-discovers which bodies have mesh variants and builds GPU caches.

    randomizer = MeshRandomizer(solver)

    print(f"\nAuto-discovered {len(randomizer.groups)} randomization groups.")

    # =====================================================================
    # Step 3: Randomize (one call per episode reset)
    # =====================================================================
    # Picks a random variant per world for each group and scatters cached
    # physics properties (mass, inertia, geom_size, etc.) to the GPU model.

    rng = np.random.default_rng(42)
    indices = randomizer.reset(solver.mjw_model, solver.mjw_data, nworld, rng)

    # =====================================================================
    # Step 4: Verify
    # =====================================================================

    # Verify: all worlds with the same variant index must have identical mass.
    mass = solver.mjw_model.body_mass.numpy()
    for gi, group in enumerate(randomizer.groups):
        bid = group.body_ids.numpy()[0]
        idx = indices[gi]
        for vi in range(group.n_variants):
            world_masses = mass[idx == vi, bid]
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
