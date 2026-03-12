# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: per-world mesh randomization.
#
# Sets model.shape_active_variant per world, calls
# notify_model_changed(MESH_VARIANT_PROPERTIES) — the solver handles the rest.
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

BOX_INDICES = np.array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
], dtype=np.int32)


def _box(hx, hy, hz, offset=(0., 0., 0.)):
    ox, oy, oz = offset
    return np.array([
        [ox - hx, oy - hy, oz - hz], [ox + hx, oy - hy, oz - hz],
        [ox + hx, oy + hy, oz - hz], [ox - hx, oy + hy, oz - hz],
        [ox - hx, oy - hy, oz + hz], [ox + hx, oy - hy, oz + hz],
        [ox + hx, oy + hy, oz + hz], [ox - hx, oy + hy, oz + hz],
    ], dtype=np.float32)


def _mesh(hx, hy, hz, offset=(0., 0., 0.)):
    return newton.Mesh(_box(hx, hy, hz, offset), BOX_INDICES)


@wp.kernel
def randomize_variants_kernel(
    sav: wp.array(dtype=wp.int32),
    shape_id: wp.array(dtype=wp.int32),
    group_id: wp.array(dtype=wp.int32),
    n_variants: wp.array(dtype=wp.int32),
    spw: int,
    seed: int,
):
    """One thread per (world, shape-slot). Writes one value to sav."""
    world, slot = wp.tid()
    grp = group_id[slot]
    rng = wp.rand_init(seed, world * 65536 + grp)
    n_var = n_variants[grp]
    variant = int(wp.randf(rng) * float(n_var))
    if variant >= n_var:
        variant = n_var - 1
    sav[shape_id[slot] + world * spw] = variant


def main():
    # ----- build one-world template -----
    wb = newton.ModelBuilder()

    body_a = wb.add_body(xform=wp.transform((0, 0, 0.5), wp.quat_identity()), label="obj_a")
    wb.add_shape_mesh(body=body_a, mesh=_mesh(0.1, 0.1, 0.1), label="geom_a",
                      mesh_variants=[_mesh(0.2, 0.2, 0.2)])

    body_b = wb.add_body(xform=wp.transform((0, 1, 0.5), wp.quat_identity()), label="obj_b")
    wb.add_shape_mesh(body=body_b, mesh=_mesh(0.15, 0.15, 0.15), label="geom_b",
                      mesh_variants=[_mesh(0.05, 0.05, 0.05)])

    body_c = wb.add_body(xform=wp.transform((0, 2, 0.5), wp.quat_identity()), label="obj_c")
    hull_x = [_box(0.08, 0.08, 0.04, offset=(i * 0.16, 0, 0)) for i in range(4)]
    hull_y = [_box(0.12, 0.06, 0.06), _box(0.12, 0.06, 0.06, (0.24, 0, 0)),
              _box(0.06, 0.12, 0.06, (0.12, 0.12, 0))]
    hull_z = [_box(0.15, 0.15, 0.08), _box(0.15, 0.15, 0.08, (0.3, 0, 0))]
    wb.add_shape_mesh(body=body_c, mesh=newton.Mesh(hull_x[0], BOX_INDICES), label="hull_0",
                      mesh_variants=[newton.Mesh(hull_y[0], BOX_INDICES), newton.Mesh(hull_z[0], BOX_INDICES)])
    wb.add_shape_mesh(body=body_c, mesh=newton.Mesh(hull_x[1], BOX_INDICES), label="hull_1",
                      mesh_variants=[newton.Mesh(hull_y[1], BOX_INDICES), newton.Mesh(hull_z[1], BOX_INDICES)])
    wb.add_shape_mesh(body=body_c, mesh=newton.Mesh(hull_x[2], BOX_INDICES), label="hull_2",
                      mesh_variants=[newton.Mesh(hull_y[2], BOX_INDICES)])
    wb.add_shape_mesh(body=body_c, mesh=newton.Mesh(hull_x[3], BOX_INDICES), label="hull_3")

    arm_base = wb.add_link(xform=wp.transform((0, 3, 0.5), wp.quat_identity()), label="arm_base")
    wb.add_shape_mesh(body=arm_base, mesh=_mesh(0.05, 0.05, 0.05), label="arm_base_geom")
    arm_l1 = wb.add_link(xform=wp.transform((0, 3, 0.8), wp.quat_identity()), label="arm_link1")
    wb.add_shape_mesh(body=arm_l1, mesh=_mesh(0.04, 0.04, 0.15), label="link1_geom",
                      mesh_variants=[_mesh(0.06, 0.06, 0.15)])
    arm_l2 = wb.add_link(xform=wp.transform((0, 3, 1.1), wp.quat_identity()), label="arm_link2")
    wb.add_shape_mesh(body=arm_l2, mesh=_mesh(0.03, 0.03, 0.12), label="link2_geom",
                      mesh_variants=[_mesh(0.05, 0.05, 0.12)])
    j0 = wb.add_joint_fixed(parent=-1, child=arm_base)
    j1 = wb.add_joint_revolute(parent=arm_base, child=arm_l1, axis=newton.Axis.Y)
    j2 = wb.add_joint_revolute(parent=arm_l1, child=arm_l2, axis=newton.Axis.Y)
    wb.add_articulation([j0, j1, j2], label="arm")

    # ----- replicate + solver -----
    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    mb.replicate(wb, world_count=NWORLD, spacing=(2, 0, 0))
    model = mb.finalize()
    solver = newton.solvers.SolverMuJoCo(model, iterations=1)
    mj_model = solver.mj_model
    nworld = solver.mjw_data.nworld

    print(f"\n{nworld} worlds, {mj_model.nbody} bodies, {mj_model.ngeom} geoms, {mj_model.nmesh} meshes")
    print(f"variant cache: {solver._has_mesh_variants}")

    # ----- discover variant groups (body → its mesh shapes that have variants) -----
    MESH_TYPES = (7, 10)  # GEO_MESH, GEO_SDF
    shape_types = model.shape_type.numpy()
    bpw = model.body_count // model.world_count
    spw = solver._shapes_per_world  # shapes per world in the replicated model

    groups = []  # (body_name, n_variants, [template_shape_indices])
    for bid in range(bpw):
        shapes = model.body_shapes.get(bid, [])
        with_var = [s for s in shapes
                    if shape_types[s] in MESH_TYPES and model.shape_mesh_variants[s]]
        if not with_var:
            continue
        all_mesh = [s for s in shapes if shape_types[s] in MESH_TYPES]
        n_var = 1 + max(len(model.shape_mesh_variants[s]) for s in with_var)
        groups.append((model.body_label[bid], n_var, all_mesh))

    print(f"{len(groups)} groups: {[(g[0], g[1]) for g in groups]}")

    # ----- build GPU lookup arrays for the randomization kernel (once) -----
    all_sids, all_gids, nvars = [], [], []
    for gi, (_, n_var, shape_ids) in enumerate(groups):
        for s in shape_ids:
            all_sids.append(s)
            all_gids.append(gi)
        nvars.append(n_var)

    dev = model.device
    n_slots = len(all_sids)
    gpu_sids = wp.array(all_sids, dtype=wp.int32, device=dev)
    gpu_gids = wp.array(all_gids, dtype=wp.int32, device=dev)
    gpu_nvars = wp.array(nvars, dtype=wp.int32, device=dev)

    # ----- randomize: one kernel launch, one thread per (world, shape-slot) -----
    wp.launch(
        randomize_variants_kernel,
        dim=(nworld, n_slots),
        inputs=[model.shape_active_variant, gpu_sids, gpu_gids,
                gpu_nvars, spw, 42],
        device=dev,
    )
    solver.notify_model_changed(SolverNotifyFlags.MESH_VARIANT_PROPERTIES)

    # ----- verify -----
    sav = model.shape_active_variant.numpy()
    mass = solver.mjw_model.body_mass.numpy()
    subtreemass = solver.mjw_model.body_subtreemass.numpy()

    for name, n_var, shape_ids in groups:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        variant_per_world = np.array([sav[shape_ids[0] + w * spw] for w in range(nworld)])
        for vi in range(n_var):
            wm = mass[variant_per_world == vi, bid]
            assert len(set(wm)) <= 1, f"{name} v{vi}: masses differ {set(wm)}"
    print("[PASS] mass consistency")

    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
    l1_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link1")
    l2_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link2")
    for w in range(nworld):
        exp = mass[w, base_id] + mass[w, l1_id] + mass[w, l2_id]
        assert abs(subtreemass[w, base_id] - exp) < 0.01
    print("[PASS] arm subtreemass")

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))
    print("[PASS] finite simulation step")


if __name__ == "__main__":
    main()
