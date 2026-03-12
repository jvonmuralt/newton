# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: per-world mesh randomization in Newton style.
#
# Uses Newton's ModelBuilder + SolverMuJoCo to create a multi-world scene,
# then demonstrates how to randomize which mesh each world uses, with
# GPU-resident caching for fast RL episode resets.
#
# Requires: mujoco_warp with PR #1191 (2D geom_dataid).
#
# Run:  uv run --extra dev python newton/tests/prototype_mesh_randomization.py

import os
import re
import tempfile

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import newton

NWORLD = 64

BOX_SMALL_VERTS = np.array([
    [-0.1, -0.1, -0.1], [0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, 0.1, -0.1],
    [-0.1, -0.1,  0.1], [0.1, -0.1,  0.1], [0.1, 0.1,  0.1], [-0.1, 0.1,  0.1],
], dtype=np.float32)

BOX_LARGE_VERTS = np.array([
    [-0.2, -0.2, -0.2], [0.2, -0.2, -0.2], [0.2, 0.2, -0.2], [-0.2, 0.2, -0.2],
    [-0.2, -0.2,  0.2], [0.2, -0.2,  0.2], [0.2, 0.2,  0.2], [-0.2, 0.2,  0.2],
], dtype=np.float32)

BOX_INDICES = np.array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
], dtype=np.int32)

# Mesh-dependent fields that need per-world values when swapping meshes.
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


# -- Warp kernels: scatter cached variant data into per-world arrays --

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


class MeshVariantCache:
    """GPU-resident cache of mesh-dependent physics properties.

    Built once at init by mini-compiling each variant via spec.compile().
    At reset, a Warp kernel scatters cached values into per-world model arrays.
    """

    def __init__(self, spec: mujoco.MjSpec, mj_model, geom_name: str,
                 mesh_names: list[str], device: str = "cuda:0"):
        self.device = device
        self.mesh_names = mesh_names
        self.n_variants = len(mesh_names)
        self.geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        self.mesh_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, mn)
            for mn in mesh_names
        ]
        self.base_dataid = mj_model.geom_dataid.copy()
        self.ngeom = mj_model.ngeom
        self.nbody = mj_model.nbody

        geom_spec = next(g for g in spec.geoms if g.name == geom_name)
        original_meshname = geom_spec.meshname
        compiled = []
        for mn in mesh_names:
            geom_spec.meshname = mn
            compiled.append(spec.compile())
        geom_spec.meshname = original_meshname

        self.cache: dict[str, wp.array] = {}
        for field, (wp_dtype, count_attr) in ALL_FIELDS.items():
            n = getattr(mj_model, count_attr)
            np_data = np.stack([getattr(ref, field) for ref in compiled], axis=0)
            self.cache[field] = wp.array(np_data, dtype=wp_dtype, device=device)

    def reset(self, mjw_model, nworld: int, rng: np.random.Generator) -> np.ndarray:
        """Randomize mesh variants across worlds."""
        local_idx = rng.integers(0, self.n_variants, size=nworld).astype(np.int32)
        variant_gpu = wp.array(local_idx, dtype=int, device=self.device)

        dataid_table = np.tile(self.base_dataid, (nworld, 1))
        for w in range(nworld):
            dataid_table[w, self.geom_id] = self.mesh_ids[local_idx[w]]
        mjw_model.geom_dataid = wp.array(dataid_table, dtype=int, device=self.device)

        for field, (wp_dtype, _) in ALL_FIELDS.items():
            n = self.ngeom if "geom" in field else self.nbody
            dst = wp.empty((nworld, n), dtype=wp_dtype, device=self.device)
            wp.launch(_SCATTER[wp_dtype], dim=(nworld, n),
                      inputs=[variant_gpu, self.cache[field], dst], device=self.device)
            setattr(mjw_model, field, dst)

        return local_idx


def main():
    # -- 1. Build Newton multi-world model --
    mesh_small = newton.Mesh(BOX_SMALL_VERTS, BOX_INDICES)

    world_builder = newton.ModelBuilder()
    body = world_builder.add_body(
        xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        label="obj",
    )
    world_builder.add_shape_mesh(body=body, mesh=mesh_small, label="obj_geom")

    main_builder = newton.ModelBuilder()
    main_builder.add_ground_plane()
    main_builder.replicate(world_builder, world_count=NWORLD, spacing=(2.0, 0.0, 0.0))

    model = main_builder.finalize()
    print(f"\n{model.world_count} worlds, {model.body_count} bodies, {model.shape_count} shapes")

    # -- 2. Create SolverMuJoCo --
    mjcf_path = os.path.join(tempfile.mkdtemp(), "newton_scene.xml")
    solver = newton.solvers.SolverMuJoCo(model, iterations=1, save_to_mjcf=mjcf_path)
    print(f"MuJoCo template: nbody={solver.mj_model.nbody}, ngeom={solver.mj_model.ngeom}, nmesh={solver.mj_model.nmesh}")

    # -- 3. Reconstruct spec, add variant mesh, recompile --
    # Strip <inertial> tags — Newton writes explicit inertial which prevents
    # the compiler from inferring mass/inertia from the swapped mesh.
    mjcf_xml = re.sub(r'\s*<inertial[^/]*/>', '', open(mjcf_path).read())
    spec = mujoco.MjSpec.from_string(mjcf_xml)
    spec.add_mesh(name="box_large", uservert=BOX_LARGE_VERTS.flatten(), userface=BOX_INDICES.flatten())
    mj_model_with_variants = spec.compile()

    geom_name = next(
        mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(mj_model_with_variants.ngeom)
        if (mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obj_geom")
    )
    mesh_names = [
        mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, i)
        for i in range(mj_model_with_variants.nmesh)
        if mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, i)
    ]
    print(f"Geom: '{geom_name}', meshes: {mesh_names}")

    # -- 4. Rebuild GPU model with variant mesh data --
    nworld = solver.mjw_data.nworld
    solver.mj_model = mj_model_with_variants
    solver.mjw_model = mujoco_warp.put_model(mj_model_with_variants)
    solver._expand_model_fields(solver.mjw_model, nworld)

    # -- 5. Build GPU cache --
    cache = MeshVariantCache(spec, mj_model_with_variants, geom_name, mesh_names)
    obj_body_id = mj_model_with_variants.geom_bodyid[cache.geom_id]

    # -- 6. Before reset: all worlds use box_small --
    mass_before = solver.mjw_model.body_mass.numpy()
    print(f"\nBefore reset: all worlds mass={mass_before[0, obj_body_id]:.1f}")

    # -- 7. After reset: worlds randomly get box_small or box_large --
    rng = np.random.default_rng(42)
    cache.reset(solver.mjw_model, nworld, rng)

    dataid = solver.mjw_model.geom_dataid.numpy()
    mass_after = solver.mjw_model.body_mass.numpy()
    for w in range(min(6, nworld)):
        mid = dataid[w, cache.geom_id]
        mn = mujoco.mj_id2name(mj_model_with_variants, mujoco.mjtObj.mjOBJ_MESH, mid)
        print(f"  world[{w}]: {mn}, mass={mass_after[w, obj_body_id]:.1f}")

    # -- 8. Step and verify finite --
    mj_data = mujoco.MjData(mj_model_with_variants)
    mujoco.mj_forward(mj_model_with_variants, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model_with_variants, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))
    print("\nSimulation step OK.")


if __name__ == "__main__":
    main()
