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
# Fields that change when a mesh is swapped
# ---------------------------------------------------------------------------

ALL_FIELDS = {
    "geom_size": (wp.vec3, "ngeom"),
    "geom_rbound": (float, "ngeom"),
    "geom_pos": (wp.vec3, "ngeom"),
    "body_mass": (float, "nbody"),
    "body_inertia": (wp.vec3, "nbody"),
    "body_ipos": (wp.vec3, "nbody"),
    "body_iquat": (wp.quat, "nbody"),
}
# body_subtreemass and body_invweight0 are tree-dependent and recomputed
# by set_const_fixed / set_const_0 after scattering.

# ---------------------------------------------------------------------------
# Warp scatter kernels  (src is compact, dst is full model array)
# ---------------------------------------------------------------------------


@wp.kernel
def _scatter_1d(idx: wp.array(dtype=int), src: wp.array2d(dtype=float),
                entity_ids: wp.array(dtype=int), dst: wp.array2d(dtype=float)):
    w, i = wp.tid()
    dst[w, entity_ids[i]] = src[idx[w], i]


@wp.kernel
def _scatter_vec2(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.vec2),
                  entity_ids: wp.array(dtype=int), dst: wp.array2d(dtype=wp.vec2)):
    w, i = wp.tid()
    dst[w, entity_ids[i]] = src[idx[w], i]


@wp.kernel
def _scatter_vec3(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.vec3),
                  entity_ids: wp.array(dtype=int), dst: wp.array2d(dtype=wp.vec3)):
    w, i = wp.tid()
    dst[w, entity_ids[i]] = src[idx[w], i]


@wp.kernel
def _scatter_quat(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.quat),
                  entity_ids: wp.array(dtype=int), dst: wp.array2d(dtype=wp.quat)):
    w, i = wp.tid()
    dst[w, entity_ids[i]] = src[idx[w], i]


_SCATTER = {
    float: _scatter_1d, wp.vec2: _scatter_vec2,
    wp.vec3: _scatter_vec3, wp.quat: _scatter_quat,
}

# ---------------------------------------------------------------------------
# MeshRandomizer — auto-discovers variant groups from solver, single reset()
# ---------------------------------------------------------------------------


class _VariantGroup:
    """One independent randomization group (one body, one or more geom slots)."""

    def __init__(self):
        self.body_name: str = ""
        self.n_variants: int = 0
        self.geom_ids: dict[str, int] = {}       # geom_name → MuJoCo geom id
        self.owned_geom_ids: wp.array = None      # (n_geoms,) on GPU
        self.owned_body_ids: wp.array = None      # (n_bodies,) on GPU
        self.dataid_rows: list[np.ndarray] = []   # per-variant geom_dataid row
        self.variants_data: dict = {}             # variant_idx → {field: np.array}
        self.gpu_cache: dict[str, wp.array] = {}  # field → (n_variants, n_owned)


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
            self._resolve_mujoco_ids(group, variants, solver.mj_model, device)
            self._build_dataid_rows(group, variants, solver.mj_model)
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
    ):
        """Map geom/body names to MuJoCo integer ids."""
        body_id_set: set[int] = set()
        for gn in variants[0]:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gn)
            group.geom_ids[gn] = gid
            body_id_set.add(int(mj_model.geom_bodyid[gid]))

        group.owned_geom_ids = wp.array(
            np.array(list(group.geom_ids.values()), dtype=np.int32),
            dtype=int, device=device,
        )
        group.owned_body_ids = wp.array(
            np.array(sorted(body_id_set), dtype=np.int32),
            dtype=int, device=device,
        )

    @staticmethod
    def _build_dataid_rows(
        group: _VariantGroup, variants: list[dict], mj_model
    ):
        """Pre-compute the geom_dataid row for each variant."""
        base = mj_model.geom_dataid.copy()
        for var in variants:
            row = base.copy()
            for gname, mname in var.items():
                gid = group.geom_ids[gname]
                row[gid] = -1 if mname is None else mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_MESH, mname
                )
            group.dataid_rows.append(row)

    # -- Compilation & caching --

    @staticmethod
    def _compute_and_cache(
        group: _VariantGroup,
        newton_model,
        mesh_shapes: list[int],
        device: str,
    ):
        """Compute physics properties directly from mesh vertices, build GPU cache.
        """
        from newton._src.geometry.inertia import compute_inertia_mesh

        n_geoms = len(mesh_shapes)
        n_bodies = group.owned_body_ids.shape[0]
        density = 1000.0  # MuJoCo default
        all_scales = newton_model.shape_scale.numpy()

        for vi in range(group.n_variants):
            # -- Per-geom properties --
            geom_sizes = np.zeros((n_geoms, 3), dtype=np.float32)
            geom_rbounds = np.zeros(n_geoms, dtype=np.float32)
            geom_positions = np.zeros((n_geoms, 3), dtype=np.float32)

            geom_masses = []
            geom_coms = []
            geom_inertias = []

            for gi, s in enumerate(mesh_shapes):
                mvs = newton_model.shape_mesh_variants[s]
                scale = all_scales[s].astype(np.float32)

                if vi == 0:
                    mesh = newton_model.shape_source[s]
                elif vi - 1 < len(mvs):
                    mesh = mvs[vi - 1]
                else:
                    # Disabled slot: zero contribution
                    geom_masses.append(0.0)
                    geom_coms.append(np.zeros(3))
                    geom_inertias.append(np.zeros((3, 3)))
                    continue

                verts = np.array(mesh.vertices, dtype=np.float32) * scale
                indices = np.array(mesh.indices, dtype=np.int32)

                mass, com, I_mat, _ = compute_inertia_mesh(density, verts, indices)
                com_np = np.array(com)
                I_mat = np.array(I_mat).reshape(3, 3)

                # geom_size = AABB half-extents
                vmin = verts.min(axis=0)
                vmax = verts.max(axis=0)
                half_extents = (vmax - vmin) / 2.0
                geom_sizes[gi] = half_extents

                # geom_rbound = bounding sphere radius from geom center
                center = (vmin + vmax) / 2.0
                geom_rbounds[gi] = np.max(np.linalg.norm(verts - center, axis=1))

                # geom_pos stays at body-frame origin (mesh defines the shape)
                geom_positions[gi] = [0.0, 0.0, 0.0]

                geom_masses.append(float(mass))
                geom_coms.append(com_np)
                geom_inertias.append(np.array(I_mat))

            # -- Body properties (combine all active geoms) --
            total_mass = sum(geom_masses)
            if total_mass > 0:
                body_com = sum(m * c for m, c in zip(geom_masses, geom_coms)) / total_mass
            else:
                body_com = np.zeros(3)

            # Combine inertia tensors at body COM using parallel axis theorem
            I_combined = np.zeros((3, 3))
            for m, c, I in zip(geom_masses, geom_coms, geom_inertias):
                if m > 0:
                    r = c - body_com
                    I_shifted = I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
                    I_combined += I_shifted

            # Diagonalize to get principal inertia + orientation
            eigenvalues, eigenvectors = np.linalg.eigh(I_combined)
            # Ensure right-handed frame
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, 0] *= -1

            from scipy.spatial.transform import Rotation
            body_iquat_scipy = Rotation.from_matrix(eigenvectors).as_quat()  # [x,y,z,w]
            body_iquat = np.array([body_iquat_scipy[3], body_iquat_scipy[0],
                                   body_iquat_scipy[1], body_iquat_scipy[2]], dtype=np.float32)

            body_inertia = eigenvalues.astype(np.float32)

            group.variants_data[vi] = {
                "geom_size": geom_sizes,
                "geom_rbound": geom_rbounds,
                "geom_pos": geom_positions,
                "body_mass": np.array([total_mass], dtype=np.float32),
                "body_inertia": np.array([body_inertia], dtype=np.float32),
                "body_ipos": np.array([body_com], dtype=np.float32),
                "body_iquat": np.array([body_iquat], dtype=np.float32),
            }

        # Stack into compact GPU arrays: shape (n_variants, n_owned)
        for field, (wp_dtype, _) in ALL_FIELDS.items():
            np_data = np.stack(
                [group.variants_data[vi][field] for vi in range(group.n_variants)],
                axis=0,
            )
            group.gpu_cache[field] = wp.array(np_data, dtype=wp_dtype, device=device)

    # -- Reset --

    def reset(
        self, mjw_model, mjw_data, nworld: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Randomize all groups independently.  Returns per-group index arrays."""
        all_indices = []
        for group in self.groups:
            local_idx = rng.integers(0, group.n_variants, size=nworld).astype(np.int32)
            variant_gpu = wp.array(local_idx, dtype=int, device=self.device)

            self._scatter_dataid(group, mjw_model, local_idx, nworld)
            self._scatter_fields(group, mjw_model, variant_gpu, nworld)

            all_indices.append(local_idx)

        # Recompute tree-dependent fields (body_subtreemass, body_invweight0)
        import mujoco_warp
        mujoco_warp.set_const_fixed(mjw_model, mjw_data)
        mujoco_warp.set_const_0(mjw_model, mjw_data)

        return all_indices

    def _scatter_dataid(
        self, group: _VariantGroup, mjw_model, local_idx: np.ndarray, nworld: int
    ):
        """Write per-world geom_dataid (which mesh each geom uses)."""
        dataid = mjw_model.geom_dataid.numpy()
        if dataid.shape[0] < nworld:
            dataid = np.tile(dataid, (nworld, 1))
        for w in range(nworld):
            row = group.dataid_rows[local_idx[w]]
            for gid in group.geom_ids.values():
                dataid[w, gid] = row[gid]
        mjw_model.geom_dataid = wp.array(dataid, dtype=int, device=self.device)

    def _scatter_fields(
        self, group: _VariantGroup, mjw_model, variant_gpu: wp.array, nworld: int
    ):
        """Scatter cached physics properties (mass, inertia, etc.) to model."""
        for field, (wp_dtype, _) in ALL_FIELDS.items():
            entity_ids = group.owned_geom_ids if "geom" in field else group.owned_body_ids
            n_owned = entity_ids.shape[0]
            dst = getattr(mjw_model, field)
            wp.launch(
                _SCATTER[wp_dtype], dim=(nworld, n_owned),
                inputs=[variant_gpu, group.gpu_cache[field], entity_ids, dst],
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

    print(f"\nAuto-discovered {len(randomizer.groups)} randomization groups:")
    for g in randomizer.groups:
        geoms = list(g.geom_ids.keys())
        print(f"  body '{g.body_name}': {g.n_variants} variants, geoms={geoms}")
        for vi, fields in g.variants_data.items():
            print(f"    variant {vi}: mass={fields['body_mass']}")

    # =====================================================================
    # Step 3: Randomize (one call per episode reset)
    # =====================================================================
    # Picks a random variant per world for each group and scatters cached
    # physics properties (mass, inertia, geom_size, etc.) to the GPU model.

    rng = np.random.default_rng(42)
    indices = randomizer.reset(solver.mjw_model, solver.mjw_data, nworld, rng)
    idx_a, idx_b, idx_c = indices

    # =====================================================================
    # Step 4: Verify
    # =====================================================================

    body_a_id = randomizer.groups[0].owned_body_ids.numpy()[0]
    body_b_id = randomizer.groups[1].owned_body_ids.numpy()[0]
    body_c_id = randomizer.groups[2].owned_body_ids.numpy()[0]

    mass = solver.mjw_model.body_mass.numpy()
    dataid = solver.mjw_model.geom_dataid.numpy()

    labels_a = ["small", "large"]
    labels_b = ["medium", "tiny"]
    labels_c = ["X (4 hulls)", "Y (3 hulls)", "Z (2 hulls)"]

    print("\nAfter reset (first 8 worlds):")
    hull_geom_ids = list(randomizer.groups[2].geom_ids.values())
    for w in range(min(8, nworld)):
        active = sum(1 for gid in hull_geom_ids if dataid[w, gid] != -1)
        print(f"  world[{w}]: "
              f"A={labels_a[idx_a[w]]}({mass[w, body_a_id]:.1f}), "
              f"B={labels_b[idx_b[w]]}({mass[w, body_b_id]:.1f}), "
              f"C={labels_c[idx_c[w]]}({mass[w, body_c_id]:.2f}, {active}/4 hulls)")

    for vi in range(2):
        masses = [mass[w, body_a_id] for w in range(nworld) if idx_a[w] == vi]
        assert all(m == masses[0] for m in masses)
    for vi in range(2):
        masses = [mass[w, body_b_id] for w in range(nworld) if idx_b[w] == vi]
        assert all(m == masses[0] for m in masses)
    for vi in range(3):
        masses = [mass[w, body_c_id] for w in range(nworld) if idx_c[w] == vi]
        if masses:
            assert all(m == masses[0] for m in masses)
    print("\nConsistency checks passed.")

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))
    print("Simulation step OK.")


if __name__ == "__main__":
    main()
