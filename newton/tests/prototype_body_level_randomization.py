# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: body-level mesh randomization (per-world).
#
# Body-level = swap ALL geoms on a body at once, as a group.
# Two variants with different geom counts:
#   Variant A (complex, 3 mesh pieces) ← default in worldbody
#   Variant B (simple, 2 mesh pieces)  ← some worlds use this
#
# By putting the most complex variant as default, no padding is needed.
# Variant B just disables the unused geom slot (dataid=-1).
#
# Requires: mujoco_warp with PR #1191 (2D geom_dataid).
#
# Run:  uv run --extra dev python newton/tests/prototype_body_level_randomization.py

import os
import re
import tempfile

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import newton

NWORLD = 64

# Variant A (complex): 3 small boxes arranged as an L-shape
PIECE_A0_VERTS = np.array([  # bottom-left
    [-0.1, -0.1, -0.05], [0.1, -0.1, -0.05], [0.1, 0.1, -0.05], [-0.1, 0.1, -0.05],
    [-0.1, -0.1,  0.05], [0.1, -0.1,  0.05], [0.1, 0.1,  0.05], [-0.1, 0.1,  0.05],
], dtype=np.float32)
PIECE_A1_VERTS = PIECE_A0_VERTS + np.array([0.2, 0.0, 0.0], dtype=np.float32)  # bottom-right
PIECE_A2_VERTS = PIECE_A0_VERTS + np.array([0.0, 0.2, 0.0], dtype=np.float32)  # top-left

# Variant B (simple): 2 larger boxes
PIECE_B0_VERTS = np.array([
    [-0.15, -0.15, -0.08], [0.15, -0.15, -0.08], [0.15, 0.15, -0.08], [-0.15, 0.15, -0.08],
    [-0.15, -0.15,  0.08], [0.15, -0.15,  0.08], [0.15, 0.15,  0.08], [-0.15, 0.15,  0.08],
], dtype=np.float32)
PIECE_B1_VERTS = PIECE_B0_VERTS + np.array([0.3, 0.0, 0.0], dtype=np.float32)

BOX_INDICES = np.array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
], dtype=np.int32)

ALL_FIELDS = {
    "geom_size": (wp.vec3, "ngeom"),
    "geom_rbound": (float, "ngeom"),
    "geom_pos": (wp.vec3, "ngeom"),
    "body_mass": (float, "nbody"),
    "body_subtreemass": (float, "nbody"),
    "body_inertia": (wp.vec3, "nbody"),
    "body_invweight0": (wp.vec2, "nbody"),
    "body_ipos": (wp.vec3, "nbody"),
    "body_iquat": (wp.quat, "nbody"),
}


@wp.kernel
def _scatter_1d(idx: wp.array(dtype=int), src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    w, e = wp.tid()
    dst[w, e] = src[idx[w], e]

@wp.kernel
def _scatter_vec2(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.vec2), dst: wp.array2d(dtype=wp.vec2)):
    w, e = wp.tid()
    dst[w, e] = src[idx[w], e]

@wp.kernel
def _scatter_vec3(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.vec3), dst: wp.array2d(dtype=wp.vec3)):
    w, e = wp.tid()
    dst[w, e] = src[idx[w], e]

@wp.kernel
def _scatter_quat(idx: wp.array(dtype=int), src: wp.array2d(dtype=wp.quat), dst: wp.array2d(dtype=wp.quat)):
    w, e = wp.tid()
    dst[w, e] = src[idx[w], e]

_SCATTER = {float: _scatter_1d, wp.vec2: _scatter_vec2, wp.vec3: _scatter_vec3, wp.quat: _scatter_quat}


class BodyVariantCache:
    """GPU-resident cache for body-level mesh randomization.

    Each variant defines a full set of mesh assignments for all geoms on a body.
    Variants with fewer geoms disable unused slots via dataid=-1.
    """

    def __init__(self, spec: mujoco.MjSpec, mj_model, body_name: str,
                 variants: list[dict[str, str | None]], device: str = "cuda:0"):
        """
        Args:
            variants: list of dicts mapping geom_name -> mesh_name (or None to disable).
                e.g. [{"g0": "a0", "g1": "a1", "g2": "a2"},   # variant A
                      {"g0": "b0", "g1": "b1", "g2": None}]   # variant B (g2 disabled)
        """
        self.device = device
        self.n_variants = len(variants)
        self.ngeom = mj_model.ngeom
        self.nbody = mj_model.nbody
        self.base_dataid = mj_model.geom_dataid.copy()

        # Resolve geom/mesh names to IDs
        self.geom_ids = {}
        for geom_name in variants[0]:
            self.geom_ids[geom_name] = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )

        self.variant_dataid_rows = []
        for var in variants:
            row = self.base_dataid.copy()
            for gname, mname in var.items():
                gid = self.geom_ids[gname]
                if mname is None:
                    row[gid] = -1
                else:
                    row[gid] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, mname)
            self.variant_dataid_rows.append(row)

        # Mini-compile each variant
        body_spec = next(b for b in spec.bodies if b.name == body_name)
        geom_specs = {g.name: g for g in body_spec.geoms if g.name in self.geom_ids}
        saved = {gn: (gs.meshname, gs.contype, gs.conaffinity) for gn, gs in geom_specs.items()}

        compiled = []
        for var in variants:
            for gname, mname in var.items():
                gs = geom_specs[gname]
                if mname is None:
                    gs.contype = 0
                    gs.conaffinity = 0
                else:
                    gs.meshname = mname
                    gs.contype = 1
                    gs.conaffinity = 1
            compiled.append(spec.compile())

        for gn, (mn, ct, ca) in saved.items():
            geom_specs[gn].meshname = mn
            geom_specs[gn].contype = ct
            geom_specs[gn].conaffinity = ca

        self.cache: dict[str, wp.array] = {}
        for field, (wp_dtype, count_attr) in ALL_FIELDS.items():
            n = getattr(mj_model, count_attr)
            np_data = np.stack([getattr(ref, field) for ref in compiled], axis=0)
            self.cache[field] = wp.array(np_data, dtype=wp_dtype, device=device)

    def reset(self, mjw_model, nworld: int, rng: np.random.Generator) -> np.ndarray:
        """Randomize body variants across worlds."""
        local_idx = rng.integers(0, self.n_variants, size=nworld).astype(np.int32)
        variant_gpu = wp.array(local_idx, dtype=int, device=self.device)

        dataid_table = np.stack(
            [self.variant_dataid_rows[local_idx[w]] for w in range(nworld)], axis=0
        )
        mjw_model.geom_dataid = wp.array(dataid_table, dtype=int, device=self.device)

        for field, (wp_dtype, _) in ALL_FIELDS.items():
            n = self.ngeom if "geom" in field else self.nbody
            dst = wp.empty((nworld, n), dtype=wp_dtype, device=self.device)
            wp.launch(_SCATTER[wp_dtype], dim=(nworld, n),
                      inputs=[variant_gpu, self.cache[field], dst], device=self.device)
            setattr(mjw_model, field, dst)

        return local_idx


def main():
    # -- 1. Build Newton model with the COMPLEX variant as default (3 geom slots) --
    mesh_a0 = newton.Mesh(PIECE_A0_VERTS, BOX_INDICES)
    mesh_a1 = newton.Mesh(PIECE_A1_VERTS, BOX_INDICES)
    mesh_a2 = newton.Mesh(PIECE_A2_VERTS, BOX_INDICES)

    world_builder = newton.ModelBuilder()
    body = world_builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        label="obj",
    )
    world_builder.add_shape_mesh(body=body, mesh=mesh_a0, label="piece_0")
    world_builder.add_shape_mesh(body=body, mesh=mesh_a1, label="piece_1")
    world_builder.add_shape_mesh(body=body, mesh=mesh_a2, label="piece_2")

    main_builder = newton.ModelBuilder()
    main_builder.add_ground_plane()
    main_builder.replicate(world_builder, world_count=NWORLD, spacing=(2.0, 0.0, 0.0))

    model = main_builder.finalize()
    print(f"\n{model.world_count} worlds, {model.body_count} bodies, {model.shape_count} shapes")

    # -- 2. Create SolverMuJoCo --
    mjcf_path = os.path.join(tempfile.mkdtemp(), "scene.xml")
    solver = newton.solvers.SolverMuJoCo(model, iterations=1, save_to_mjcf=mjcf_path)
    print(f"MuJoCo template: nbody={solver.mj_model.nbody}, ngeom={solver.mj_model.ngeom}, nmesh={solver.mj_model.nmesh}")

    # -- 3. Reconstruct spec, add variant B meshes --
    mjcf_xml = re.sub(r'\s*<inertial[^/]*/>', '', open(mjcf_path).read())
    spec = mujoco.MjSpec.from_string(mjcf_xml)
    spec.add_mesh(name="piece_b0", uservert=PIECE_B0_VERTS.flatten(), userface=BOX_INDICES.flatten())
    spec.add_mesh(name="piece_b1", uservert=PIECE_B1_VERTS.flatten(), userface=BOX_INDICES.flatten())
    mj_model_with_variants = spec.compile()

    # Discover geom names on the body
    geom_names = []
    for i in range(mj_model_with_variants.ngeom):
        gn = mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_GEOM, i)
        if gn and gn.startswith("piece_"):
            geom_names.append(gn)
    print(f"Body geoms: {geom_names}")

    all_meshes = [
        mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, i)
        for i in range(mj_model_with_variants.nmesh)
        if mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, i)
    ]
    print(f"All meshes: {all_meshes}")

    # -- 4. Rebuild GPU model --
    nworld = solver.mjw_data.nworld
    solver.mj_model = mj_model_with_variants
    solver.mjw_model = mujoco_warp.put_model(mj_model_with_variants)
    solver._expand_model_fields(solver.mjw_model, nworld)

    # -- 5. Define body variants --
    # Variant A (complex, 3 pieces): default meshes
    # Variant B (simple, 2 pieces): different meshes, slot 2 disabled
    variant_a = {geom_names[0]: all_meshes[0], geom_names[1]: all_meshes[1], geom_names[2]: all_meshes[2]}
    variant_b = {geom_names[0]: "piece_b0", geom_names[1]: "piece_b1", geom_names[2]: None}

    cache = BodyVariantCache(
        spec, mj_model_with_variants, "obj",
        variants=[variant_a, variant_b],
    )

    # -- 6. Before reset --
    mass_before = solver.mjw_model.body_mass.numpy()
    obj_body_id = 1
    print(f"\nBefore reset: all worlds mass={mass_before[0, obj_body_id]:.2f} (3 pieces)")

    # -- 7. After reset --
    rng = np.random.default_rng(42)
    local_idx = cache.reset(solver.mjw_model, nworld, rng)

    dataid = solver.mjw_model.geom_dataid.numpy()
    mass_after = solver.mjw_model.body_mass.numpy()
    for w in range(min(6, nworld)):
        variant = "A (3 pieces)" if local_idx[w] == 0 else "B (2 pieces)"
        geom_meshes = []
        for gn in geom_names:
            gid = mujoco.mj_name2id(mj_model_with_variants, mujoco.mjtObj.mjOBJ_GEOM, gn)
            mid = dataid[w, gid]
            if mid == -1:
                geom_meshes.append("disabled")
            else:
                geom_meshes.append(mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, mid))
        print(f"  world[{w}]: {variant}, mass={mass_after[w, obj_body_id]:.2f}, geoms={geom_meshes}")

    # -- 8. Step and verify --
    mj_data = mujoco.MjData(mj_model_with_variants)
    mujoco.mj_forward(mj_model_with_variants, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model_with_variants, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))
    print("\nSimulation step OK.")


if __name__ == "__main__":
    main()
