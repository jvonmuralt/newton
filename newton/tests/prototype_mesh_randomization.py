# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Prototype: per-world mesh randomization using the mesh pool API.
#
# User workflow:
#   1. Register meshes in the pool via builder.add_mesh()
#   2. Create shapes that reference pool meshes
#   3. At runtime, write new mesh IDs into model.shape_mesh_id
#   4. Recompute body mass/com/inertia for affected bodies
#   5. Call solver.notify_model_changed(SHAPE_MESH_PROPERTIES | BODY_INERTIAL_PROPERTIES)
#
# The solver only syncs geom_dataid/geom_rbound — mass is the user's job.
#
# Requires: mujoco_warp with PR #1191 (2D geom_dataid).
#
# Run:  python newton/tests/prototype_mesh_randomization.py

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

import newton
from newton._src.geometry.inertia import compute_inertia_mesh
from newton._src.solvers.flags import SolverNotifyFlags

NWORLD = 64
DENSITY = 1000.0

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


def compute_body_inertia(meshes, scales, density):
    """Compute aggregated (mass, com, inertia_3x3) for a body from its mesh shapes."""
    masses, coms, inertias = [], [], []
    for mesh, scale in zip(meshes, scales):
        verts = np.asarray(mesh.vertices, dtype=np.float32) * scale
        indices = np.asarray(mesh.indices, dtype=np.int32)
        m, c, I, _ = compute_inertia_mesh(density, verts, indices)
        masses.append(float(m))
        coms.append(np.asarray(c))
        inertias.append(np.asarray(I).reshape(3, 3))

    total_mass = sum(masses)
    if total_mass <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(m * c for m, c in zip(masses, coms)) / total_mass
    combined_I = np.zeros((3, 3))
    for m, c, I in zip(masses, coms, inertias):
        if m > 0:
            r = c - com
            combined_I += I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return total_mass, com, combined_I


def main():
    # ===== 1. Build model: register meshes, create bodies =====

    wb = newton.ModelBuilder()

    # Mesh pool — every mesh the simulation might use at runtime.
    mesh_a_small = wb.add_mesh(_mesh(0.1, 0.1, 0.1))    # pool id 0
    mesh_a_big   = wb.add_mesh(_mesh(0.2, 0.2, 0.2))    # pool id 1
    mesh_b_large = wb.add_mesh(_mesh(0.15, 0.15, 0.15))  # pool id 2
    mesh_b_tiny  = wb.add_mesh(_mesh(0.05, 0.05, 0.05))  # pool id 3

    arm_mesh_base      = wb.add_mesh(_mesh(0.05, 0.05, 0.05))
    arm_mesh_l1_thin   = wb.add_mesh(_mesh(0.04, 0.04, 0.15))
    arm_mesh_l1_thick  = wb.add_mesh(_mesh(0.06, 0.06, 0.15))
    arm_mesh_l2_thin   = wb.add_mesh(_mesh(0.03, 0.03, 0.12))
    arm_mesh_l2_thick  = wb.add_mesh(_mesh(0.05, 0.05, 0.12))

    hull_meshes = [
        wb.add_mesh(newton.Mesh(_box(0.08, 0.08, 0.04, offset=(i * 0.16, 0, 0)), BOX_INDICES))
        for i in range(4)
    ]
    hull_simple = wb.add_mesh(_mesh(0.3, 0.08, 0.04))  # single box replacing the 4-piece decomposition

    # Bodies and shapes — each shape starts with a default mesh from the pool.
    body_a = wb.add_body(xform=wp.transform((0, 0, 0.5), wp.quat_identity()), label="obj_a")
    wb.add_shape_mesh(body=body_a, mesh=mesh_a_small, label="geom_a")

    body_b = wb.add_body(xform=wp.transform((0, 1, 0.5), wp.quat_identity()), label="obj_b")
    wb.add_shape_mesh(body=body_b, mesh=mesh_b_large, label="geom_b")

    body_c = wb.add_body(xform=wp.transform((0, 2, 0.5), wp.quat_identity()), label="obj_c")
    for i, mid in enumerate(hull_meshes):
        wb.add_shape_mesh(body=body_c, mesh=mid, label=f"hull_{i}")

    arm_base = wb.add_link(xform=wp.transform((0, 3, 0.5), wp.quat_identity()), label="arm_base")
    wb.add_shape_mesh(body=arm_base, mesh=arm_mesh_base, label="arm_base_geom")
    arm_l1 = wb.add_link(xform=wp.transform((0, 3, 0.8), wp.quat_identity()), label="arm_link1")
    wb.add_shape_mesh(body=arm_l1, mesh=arm_mesh_l1_thin, label="link1_geom")
    arm_l2 = wb.add_link(xform=wp.transform((0, 3, 1.1), wp.quat_identity()), label="arm_link2")
    wb.add_shape_mesh(body=arm_l2, mesh=arm_mesh_l2_thin, label="link2_geom")
    j0 = wb.add_joint_fixed(parent=-1, child=arm_base)
    j1 = wb.add_joint_revolute(parent=arm_base, child=arm_l1, axis=newton.Axis.Y)
    j2 = wb.add_joint_revolute(parent=arm_l1, child=arm_l2, axis=newton.Axis.Y)
    wb.add_articulation([j0, j1, j2], label="arm")

    # ===== 2. Replicate + create solver =====

    mb = newton.ModelBuilder()
    mb.add_ground_plane()
    mb.replicate(wb, world_count=NWORLD, spacing=(2, 0, 0))
    model = mb.finalize()
    solver = newton.solvers.SolverMuJoCo(model, iterations=1)
    mj_model = solver.mj_model
    nworld = solver.mjw_data.nworld
    shapes_per_world = solver._shapes_per_world
    bodies_per_world = model.body_count // model.world_count

    print(f"\n{nworld} worlds, {mj_model.nbody} mj bodies, {mj_model.ngeom} mj geoms, {mj_model.nmesh} mj meshes")
    print(f"mesh pool: {len(model.meshes)} meshes, shapes_per_world: {shapes_per_world}")
    print(f"mesh pool registered: {solver._mesh_to_dataid is not None}")

    # ===== 3. Define what can be swapped =====
    #
    # Case 1 (big/small): single shape, pick one mesh from a list.
    # Case 2 (decomposition): multi-shape body, swap between complex (4 hulls)
    #         and simple (1 mesh + 3 disabled via -1).
    # Case 3 (kinematic chain): shapes on articulated links.

    swappable = [
        ("obj_a",     "geom_a",     [mesh_a_small, mesh_a_big]),
        ("obj_b",     "geom_b",     [mesh_b_large, mesh_b_tiny]),
        ("arm_link1", "link1_geom", [arm_mesh_l1_thin, arm_mesh_l1_thick]),
        ("arm_link2", "link2_geom", [arm_mesh_l2_thin, arm_mesh_l2_thick]),
    ]

    # Decomposition swap: obj_c has 4 hull shapes.
    # "complex" = original 4 hulls, "simple" = hull_0 becomes hull_simple, rest disabled.
    decomp_shapes = ["hull_0", "hull_1", "hull_2", "hull_3"]
    decomp_complex = [hull_meshes[0], hull_meshes[1], hull_meshes[2], hull_meshes[3]]
    decomp_simple  = [hull_simple,    -1,             -1,             -1]

    # Resolve labels to template (world-0) indices — take first occurrence only.
    shape_label_to_idx = {}
    for i, label in enumerate(model.shape_label):
        if label not in shape_label_to_idx:
            shape_label_to_idx[label] = i
    body_label_to_idx = {}
    for i, label in enumerate(model.body_label):
        if label not in body_label_to_idx:
            body_label_to_idx[label] = i

    print(f"{len(swappable)} swappable shapes + 1 decomposition group")

    # ===== 4. Randomize shape_mesh_id per world =====

    rng = np.random.default_rng(42)
    mesh_ids = model.shape_mesh_id.numpy()
    shape_scales = model.shape_scale.numpy()

    # Case 1 & 3: simple per-shape randomization.
    for body_label, shape_label, mesh_options in swappable:
        shape_idx = shape_label_to_idx[shape_label]
        choices = rng.integers(0, len(mesh_options), size=nworld)
        for w in range(nworld):
            mesh_ids[shape_idx + w * shapes_per_world] = mesh_options[choices[w]]

    # Case 2: decomposition — randomly pick complex vs simple per world.
    decomp_use_simple = rng.integers(0, 2, size=nworld).astype(bool)
    for si, shape_label in enumerate(decomp_shapes):
        shape_idx = shape_label_to_idx[shape_label]
        for w in range(nworld):
            mesh_ids[shape_idx + w * shapes_per_world] = (
                decomp_simple[si] if decomp_use_simple[w] else decomp_complex[si]
            )

    model.shape_mesh_id.assign(wp.array(mesh_ids, dtype=wp.int32, device=model.device))

    # ===== 5. Recompute body mass/com/inertia for swapped bodies =====

    body_mass_np = model.body_mass.numpy()
    body_com_np = model.body_com.numpy()
    body_inertia_np = model.body_inertia.numpy()

    # Cases 1 & 3: single-shape bodies.
    for body_label, shape_label, _ in swappable:
        body_idx_template = body_label_to_idx[body_label]
        shape_idx = shape_label_to_idx[shape_label]

        for w in range(nworld):
            body_idx = body_idx_template + w * bodies_per_world
            global_shape = shape_idx + w * shapes_per_world
            mid = mesh_ids[global_shape]
            mesh = model.meshes[mid]
            scale = shape_scales[shape_idx].astype(np.float32)

            mass, com, inertia = compute_body_inertia([mesh], [scale], DENSITY)
            body_mass_np[body_idx] = mass
            body_com_np[body_idx] = com
            body_inertia_np[body_idx] = inertia

    # Case 2: obj_c multi-shape body — aggregate active shapes only.
    body_c_template = body_label_to_idx["obj_c"]
    for w in range(nworld):
        body_idx = body_c_template + w * bodies_per_world
        active_meshes, active_scales = [], []
        for shape_label in decomp_shapes:
            shape_idx = shape_label_to_idx[shape_label]
            mid = mesh_ids[shape_idx + w * shapes_per_world]
            if mid >= 0:
                active_meshes.append(model.meshes[mid])
                active_scales.append(shape_scales[shape_idx].astype(np.float32))
        mass, com, inertia = compute_body_inertia(active_meshes, active_scales, DENSITY)
        body_mass_np[body_idx] = mass
        body_com_np[body_idx] = com
        body_inertia_np[body_idx] = inertia

    model.body_mass.assign(wp.array(body_mass_np, dtype=wp.float32, device=model.device))
    model.body_com.assign(wp.array(body_com_np, dtype=wp.vec3, device=model.device))
    model.body_inertia.assign(wp.array(body_inertia_np, dtype=wp.mat33, device=model.device))

    # ===== 6. Notify solver =====

    solver.notify_model_changed(
        SolverNotifyFlags.SHAPE_MESH_PROPERTIES | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES,
    )

    # ===== 7. Verify =====

    print("\n--- verification ---")

    mass_mj = solver.mjw_model.body_mass.numpy()
    subtreemass_mj = solver.mjw_model.body_subtreemass.numpy()

    # Mass consistency: recompute expected mass from the assigned mesh and compare.
    for body_label, shape_label, _ in swappable:
        bid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_label)
        shape_idx = shape_label_to_idx[shape_label]
        scale = shape_scales[shape_idx].astype(np.float32)
        for w in range(nworld):
            mid = mesh_ids[shape_idx + w * shapes_per_world]
            exp_mass, _, _ = compute_body_inertia([model.meshes[mid]], [scale], DENSITY)
            assert abs(mass_mj[w, bid_mj] - exp_mass) < 0.01, \
                f"{body_label} w{w}: expected {exp_mass:.4f}, got {mass_mj[w, bid_mj]:.4f}"
    print("[PASS] mass consistency")

    # Decomposition mass: obj_c mass should match active shapes only.
    bid_c_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "obj_c")
    for w in range(nworld):
        active_meshes, active_scales = [], []
        for shape_label in decomp_shapes:
            shape_idx = shape_label_to_idx[shape_label]
            mid = mesh_ids[shape_idx + w * shapes_per_world]
            if mid >= 0:
                active_meshes.append(model.meshes[mid])
                active_scales.append(shape_scales[shape_idx].astype(np.float32))
        exp_mass, _, _ = compute_body_inertia(active_meshes, active_scales, DENSITY)
        assert abs(mass_mj[w, bid_c_mj] - exp_mass) < 0.01, \
            f"obj_c w{w}: expected {exp_mass:.4f}, got {mass_mj[w, bid_c_mj]:.4f}"
    n_simple = int(decomp_use_simple.sum())
    print(f"[PASS] decomposition mass ({n_simple}/{nworld} simple, {nworld - n_simple}/{nworld} complex)")

    # Subtreemass for articulated arm: base = base + link1 + link2.
    base_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_base")
    l1_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link1")
    l2_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_link2")
    for w in range(nworld):
        exp = mass_mj[w, base_mj] + mass_mj[w, l1_mj] + mass_mj[w, l2_mj]
        assert abs(subtreemass_mj[w, base_mj] - exp) < 0.01
    print("[PASS] arm subtreemass")

    # Simulation step produces finite values.
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    solver.mjw_data = mujoco_warp.put_data(mj_model, mj_data, nworld=nworld)
    mujoco_warp.step(solver.mjw_model, solver.mjw_data)
    assert np.all(np.isfinite(solver.mjw_data.qpos.numpy()))
    print("[PASS] finite simulation step")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
