# Newton Determinism Test Harness

A small framework for asserting and visualizing rigid-body-solver determinism
in Newton. Built specifically to help investigate how Warp's new
`wp.config.deterministic` mode interacts with Newton's existing solvers,
collision pipeline, and differentiable simulation gradients.

## Layout

```
scripts/
├── run_determinism.py              # CLI entry point (thin)
└── determinism/
    ├── __init__.py                 # Public re-exports
    ├── harness.py                  # Scenario base class, SolverSpec, snapshot, compare_runs
    └── scenarios/
        ├── __init__.py             # Scenario registry
        ├── falling_cube.py         # 1. Single cube onto floor
        ├── box_stack.py            # 2. Stack of 20 boxes
        ├── domino_chain.py         # 3. Toppling domino row
        ├── arm_7dof.py             # 4. Franka Panda PD tracking
        ├── humanoid.py             # 5. Unitree H1 standing
        ├── diffsim_ball.py         # 6. Particle target gradient with contacts
        ├── diffsim_cloth_com.py    # 7. Cloth COM gradient with atomic accumulation
        ├── diffsim_spring_cage.py  # 8. Spring rest-length gradient
        ├── vbd_cloth_patch.py      # 9. VBD cloth contact patch
        ├── vbd_soft_body.py        # 10. VBD tetrahedral soft body
        └── warp_*.py               # Focused Warp deterministic-lowering probes
```

## Quick start

To test against a sibling Warp checkout without committing a Newton dependency
override, install it into the uv venv and run with `--no-sync`:

```bash
uv pip install -e ../warp
uv run --no-sync python -c "import warp as wp; print(wp.__version__, wp.__file__)"
```

Plain `uv run` may re-sync the locked `warp-lang` wheel from `uv.lock`.

```bash
# See what's available
uv run --no-sync python scripts/run_determinism.py --list

# Single-process headless run of one scenario
uv run --no-sync python scripts/run_determinism.py \
    --scenario falling_cube --solver xpbd --world-count 16 --num-steps 500 \
    --viewer null

# Check determinism across 3 independent subprocess invocations
uv run --no-sync python scripts/run_determinism.py \
    --scenario falling_cube --solver xpbd --runs 3 --print-extras

# Visualize interactively (opens an OpenGL window)
uv run --no-sync python scripts/run_determinism.py \
    --scenario domino_chain --solver xpbd --world-count 1 \
    --num-steps 600 --viewer gl

# Toggle Warp's new deterministic atomics
uv run --no-sync python scripts/run_determinism.py \
    --scenario box_stack --solver xpbd --runs 3 \
    --warp-deterministic run_to_run

# MuJoCo determinism (the constructor pins deterministic_max_records=16).
uv run --no-sync python scripts/run_determinism.py \
    --scenario box_stack --solver mujoco --runs 3 \
    --warp-deterministic run_to_run --solver-deterministic run_to_run
```

## Scenario × solver matrix

| # | Scenario          | xpbd | featherstone | semi_implicit | vbd | mujoco |
|---|-------------------|------|--------------|---------------|-----|--------|
| 1 | falling_cube      | yes  | yes          | yes           | yes | yes    |
| 2 | box_stack         | yes  |              |               | yes | yes    |
| 3 | domino_chain      | yes  |              |               | yes | yes    |
| 4 | arm_7dof          | yes  | yes          |               |     | yes    |
| 5 | humanoid          | yes  | yes          |               |     | yes    |
| 6 | diffsim_ball      |      |              | yes           |     |        |
| 7 | diffsim_cloth_com |      |              | yes           |     |        |
| 8 | diffsim_spring_cage |    |              | yes           |     |        |
| 9 | vbd_cloth_patch   |      |              |               | yes |        |
|10 | vbd_soft_body     |      |              |               | yes |        |

The `warp_*` scenarios are headless micro-workloads rather than physics
scenes. They use the `xpbd` solver selector only to fit the existing CLI,
but they do not construct a Newton solver. Run them with `--viewer null`.

| Scenario                    | Pattern exposed |
|-----------------------------|-----------------|
| `warp_counter_indexed`      | Consumed-return `atomic_add` with a dynamic counter index, like per-body or per-world slot allocation. |
| `warp_counter_static_index` | Consumed-return `atomic_add` with a nonzero `wp.static()` index. |
| `warp_counter_sliced`       | Consumed-return `atomic_add` through a sliced per-world counter view. |
| `warp_custom_adjoint`       | Custom `@wp.func_grad` accumulation through `wp.adjoint[array]`. |
| `warp_argmax_exchange`      | Consumed-return `atomic_max` followed by an `atomic_exch` winner update. |

Unsupported scenario/solver pairs are rejected at startup with a clear
error, not silently skipped.

## How determinism is checked

Every physics `Scenario.snapshot()` returns a `ScenarioSnapshot` with two fields:

- `core`: `body_q`, `body_qd`, `joint_q`, `joint_qd` — always present, and
  the *canonical* byte-level determinism fingerprint.
- `extras`: scenario-specific telemetry (contact counts, COM, torques, …).
  Reported for diagnostics but not required to be bit-exact.

The `warp_*` micro-scenarios put their small output arrays directly in
`core` so `--runs >= 2` compares the isolated lowering workload itself.
The `diffsim_*` scenarios do the same for final particle state, loss, and
gradient arrays.

When `--runs >= 2`, the CLI re-invokes itself in N independent subprocesses
with identical args, computes a SHA-256 of each snapshot's `core` bytes,
and asserts all hashes match. Process boundaries guarantee no hidden
Python-level state leaks between runs.

## Adding a new scenario

1. Subclass `Scenario` in a new file under `scripts/determinism/scenarios/`.
2. Implement `build_subworld(builder)` to populate a single-world builder.
   The harness replicates it across `args.world_count` worlds and adds the
   ground plane. Do NOT call `finalize()` or `add_ground_plane()` yourself.
3. Optionally override:
   - `_on_built()` for one-time post-build setup (buffers, histories),
   - `per_step()` for control signals applied each physics substep
     (graph-captured on CUDA — keep it pure Warp),
   - `step()` for per-frame host-side logging (runs outside the captured
     graph so `.numpy()` etc. are cheap),
   - `extra_snapshot()` for scenario-specific telemetry merged into
     `ScenarioSnapshot.extras`.
4. Register the class in `scenarios/__init__.py`.

## Known limitations

- `SolverMuJoCo` requires cuSolverDx ≥ CUDA Toolkit 12.6.3 (build Warp
  against 12.8 or later).
- Some MJWarp solver kernels emit data-dependent deterministic atomic records.
  `SolverMuJoCo` pins `deterministic_max_records=16` in its constructor so the
  harness gets a sensible default; tweak it directly in code for stress cases.
- The `warp_*` micro-scenarios target specific deterministic-lowering
  patterns. With the determinism Warp branch, they are expected to be
  bit-exact under `--warp-deterministic run_to_run` except for
  `warp_argmax_exchange`, which currently reports Warp's explicit
  consumed-return `atomic_max` limitation.
- The report scenarios currently pass bit-exact under
  `--warp-deterministic run_to_run`; the `warp_*` micro-scenarios are separate
  lowering probes and are not part of the solver matrix.
- `ViewerGL` opens a blocking window; use `--viewer null` for any CI /
  automated runs.
