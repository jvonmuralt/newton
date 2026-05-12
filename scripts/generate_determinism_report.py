# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a static determinism report with ViewerGL videos.

The report is intentionally self-contained: it writes MP4 videos, poster PNGs,
JSON results, and a static HTML page under ``scripts/determinism/report``.
Run it from the repository root:

    uv run --with imageio --with imageio-ffmpeg python scripts/generate_determinism_report.py
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUT_DIR = SCRIPTS_DIR / "determinism" / "report"

REPRESENTATIVE_SOLVERS = {
    "falling_cube": "xpbd",
    "box_stack": "xpbd",
    "domino_chain": "xpbd",
    "arm_7dof": "featherstone",
    "humanoid": "featherstone",
    "diffsim_ball": "semi_implicit",
    "diffsim_cloth_com": "semi_implicit",
    "diffsim_spring_cage": "semi_implicit",
    "vbd_cloth_patch": "vbd",
    "vbd_soft_body": "vbd",
}

CAMERAS = {
    "falling_cube": ((2.0, -3.0, 2.0), -25.0, 125.0, 50.0),
    "box_stack": ((2.4, -3.8, 2.6), -22.0, 130.0, 48.0),
    "domino_chain": ((1.15, -2.2, 1.1), -15.0, 105.0, 52.0),
    "arm_7dof": ((1.6, -1.8, 1.15), -20.0, 135.0, 54.0),
    "humanoid": ((3.0, -4.0, 2.0), -18.0, 130.0, 52.0),
    "diffsim_ball": ((2.5, -4.0, 2.2), -24.0, 125.0, 52.0),
    "diffsim_cloth_com": ((2.8, -3.8, 2.0), -22.0, 126.0, 52.0),
    "diffsim_spring_cage": ((2.1, -2.8, 1.6), -24.0, 128.0, 50.0),
    "vbd_cloth_patch": ((1.2, -1.8, 1.1), -18.0, 130.0, 52.0),
    "vbd_soft_body": ((1.0, -1.6, 1.0), -18.0, 130.0, 52.0),
}

SCENARIO_TITLES = {
    "falling_cube": "Falling Cube",
    "box_stack": "Box Stack",
    "domino_chain": "Domino Chain",
    "arm_7dof": "7-DOF Arm",
    "humanoid": "Humanoid Standing",
    "diffsim_ball": "Diffsim Ball",
    "diffsim_cloth_com": "Diffsim Cloth COM",
    "diffsim_spring_cage": "Diffsim Spring Cage",
    "vbd_cloth_patch": "VBD Cloth Patch",
    "vbd_soft_body": "VBD Soft Body",
}

SCENARIO_NOTES = {
    "falling_cube": "Single free rigid body floor contact. Useful for detecting tiny contact drift after impact.",
    "box_stack": "Twenty jittered boxes with many contacts. A compact stress case for ordering and reductions.",
    "domino_chain": "Sequential impacts amplify small timing changes into visible propagation differences.",
    "arm_7dof": "Franka arm tracks deterministic sinusoidal PD targets from a fixed base.",
    "humanoid": "Unitree H1 holds a standing pose; drift or falling exposes articulated-contact robustness issues.",
    "diffsim_ball": "Differentiable particle target with wall and floor contacts; core includes loss and initial-velocity gradients.",
    "diffsim_cloth_com": "Small differentiable cloth target; COM uses an atomic accumulation before backpropagating gradients.",
    "diffsim_spring_cage": "Differentiable spring-cage target; core includes loss and spring rest-length gradients.",
    "vbd_cloth_patch": "Pinned cloth patch with ground contact, using VBD bending-aware coloring.",
    "vbd_soft_body": "Pinned tetrahedral block with ground contact, using VBD particle/tet solve paths.",
}


@dataclasses.dataclass
class RunConfig:
    """Shared configuration for report determinism runs and videos."""

    runs: int
    num_steps: int
    substeps: int
    fps: int
    world_count: int
    seed: int
    warp_deterministic: str


@dataclasses.dataclass(frozen=True)
class ReportVariant:
    """One report column / solver variant to benchmark."""

    id: str
    label: str
    sort_order: int
    collision_pipeline_deterministic: bool = False
    collision_pipeline_warp_deterministic: str | None = None


DEFAULT_REPORT_VARIANT = ReportVariant(
    id="default",
    label="Default",
    sort_order=0,
)

COLLISION_PIPELINE_WARP_OFF_VARIANT = ReportVariant(
    id="collision_pipeline_sort_only",
    label="CollisionPipeline deterministic=True; collide Warp off",
    sort_order=1,
    collision_pipeline_deterministic=True,
    collision_pipeline_warp_deterministic="not_guaranteed",
)


def _json_default(value: Any) -> Any:
    """Convert NumPy values into JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _short_doc(cls: type) -> str:
    doc = (cls.__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""


def _format_float(value: float, precision: int = 3) -> str:
    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def _safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _humanize_mode(mode: str) -> str:
    return mode.replace("_", " ")


def _ordered_scenarios(scenarios: dict[str, Any], xpbd_only: bool) -> list[str]:
    ordered = [scenario for scenario in SCENARIO_TITLES if scenario in scenarios]
    if xpbd_only:
        ordered = [scenario for scenario in ordered if "xpbd" in scenarios[scenario].supported_solvers]
    return ordered


def _representative_solver(scenario: str, scen_cls: type, xpbd_only: bool) -> str:
    if xpbd_only and "xpbd" in scen_cls.supported_solvers:
        return "xpbd"
    preferred = REPRESENTATIVE_SOLVERS.get(scenario)
    if preferred in scen_cls.supported_solvers:
        return preferred
    return scen_cls.supported_solvers[0]


def _report_variants() -> list[ReportVariant]:
    return [DEFAULT_REPORT_VARIANT]


def _dominant_diff(diff_map: dict[str, float]) -> tuple[str | None, float]:
    if not diff_map:
        return None, 0.0
    key, value = max(diff_map.items(), key=lambda item: abs(float(item[1])))
    return key, float(value)


def _hash_core(snapshot: Any) -> str:
    from scripts.determinism import hash_core

    return hash_core(snapshot)


def _write_snapshot(path: Path, snapshot: Any) -> None:
    from scripts.determinism import write_snapshot_to_path

    write_snapshot_to_path(snapshot, path)


def _max_core_diff(snapshots: list[Any]) -> dict[str, float]:
    ref = snapshots[0]
    max_by_key: dict[str, float] = {}
    for key, ref_arr in ref.core.items():
        max_diff = 0.0
        ref_num = np.asarray(ref_arr, dtype=np.float64)
        for snap in snapshots[1:]:
            other = np.asarray(snap.core[key], dtype=np.float64)
            if other.shape == ref_num.shape:
                max_diff = max(max_diff, float(np.max(np.abs(other - ref_num))))
        max_by_key[key] = max_diff
    return max_by_key


def _extras_diff(snapshots: list[Any]) -> dict[str, float]:
    ref = snapshots[0]
    out: dict[str, float] = {}
    for key, value in ref.extras.items():
        arr = np.asarray(value)
        if arr.dtype.kind not in "fiub" or arr.shape == ():
            continue
        max_diff = 0.0
        for snap in snapshots[1:]:
            if key not in snap.extras:
                continue
            other = np.asarray(snap.extras[key])
            if other.shape != arr.shape or other.dtype.kind not in "fiub":
                continue
            max_diff = max(
                max_diff,
                float(np.max(np.abs(other.astype(np.float64) - arr.astype(np.float64)))),
            )
        out[key] = max_diff
    return out


def _metric(label: str, value: Any, unit: str = "") -> dict[str, str]:
    if isinstance(value, (float, np.floating)):
        text = _format_float(float(value))
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        text = str(value)
    if unit:
        text = f"{text} {unit}"
    return {"label": label, "value": text}


def _result_summary(result: dict[str, Any]) -> str:
    if result["status"] == "failed":
        tail = (result.get("error") or "").splitlines()
        if tail:
            return f"Subprocess failed before a verdict could be produced: {tail[-1].strip()}"
        return "Subprocess failed before a verdict could be produced."

    run_count = max(1, len(result.get("hashes", [])))
    graph = "no graph" if result.get("disable_graph") else "CUDA graph"
    if result["status"] == "bit_exact":
        return f"{run_count}/{run_count} subprocess replays matched byte-for-byte with {graph} enabled."

    parts = [f"{result['unique_hashes']} unique hashes across {run_count} subprocess runs"]
    dominant_core = result.get("dominant_core_signal")
    dominant_core_value = float(result.get("dominant_core_value", 0.0))
    if dominant_core is not None:
        parts.append(f"largest core drift in {dominant_core} ({_format_float(dominant_core_value)})")
    dominant_extra = result.get("dominant_extra_signal")
    dominant_extra_value = float(result.get("dominant_extra_value", 0.0))
    if dominant_extra is not None and dominant_extra_value > 0.0:
        parts.append(f"largest diagnostic drift in {dominant_extra} ({_format_float(dominant_extra_value)})")
    parts.append(graph)
    return "; ".join(parts) + "."


def _build_analysis(
    solver_results: list[dict[str, Any]],
    runs: int,
    xpbd_only: bool,
    warp_deterministic: str,
) -> dict[str, Any]:
    drift = [result for result in solver_results if result["status"] == "non_deterministic"]
    failed = [result for result in solver_results if result["status"] == "failed"]

    findings: list[dict[str, str]] = []
    findings.append(
        {
            "tone": "ok" if not failed else "warn",
            "title": "Replay health",
            "body": (
                f"All {len(solver_results)} solver pairs completed {runs} independent subprocess replays."
                if not failed
                else f"{len(failed)} solver pair(s) failed before a final verdict could be computed."
            ),
        }
    )
    findings.append(
        {
            "tone": "ok" if not drift else "warn",
            "title": "Run-to-run verdict",
            "body": (
                f"Every tested solver pair was bit-exact under wp.config.deterministic={warp_deterministic!r}."
                if not drift
                else f"{len(drift)} of {len(solver_results)} solver pair(s) still diverged under "
                f"wp.config.deterministic={warp_deterministic!r}."
            ),
        }
    )
    collision_variant = [
        result for result in solver_results if result.get("variant_id") == COLLISION_PIPELINE_WARP_OFF_VARIANT.id
    ]
    if collision_variant:
        collision_variant_drift = [result for result in collision_variant if result["status"] != "bit_exact"]
        findings.append(
            {
                "tone": "ok" if not collision_variant_drift else "warn",
                "title": "CollisionPipeline deterministic=True check",
                "body": (
                    "XPBD stayed bit-exact when collide kernels were compiled with "
                    "wp.config.deterministic='not_guaranteed' and only "
                    "CollisionPipeline deterministic sorting was enabled."
                    if not collision_variant_drift
                    else f"{len(collision_variant_drift)} XPBD scenario(s) still drift when only "
                    "CollisionPipeline deterministic sorting is enabled and collide kernels use "
                    "wp.config.deterministic='not_guaranteed'."
                ),
            }
        )
    if drift:
        top = max(
            drift,
            key=lambda result: (
                float(result.get("max_core_diff_overall", 0.0)),
                float(result.get("max_extras_diff_overall", 0.0)),
            ),
        )
        findings.append(
            {
                "tone": "warn",
                "title": "Largest remaining gap",
                "body": (
                    f"{SCENARIO_TITLES.get(top['scenario'], top['scenario'])} / {top['solver']} drifts the most: "
                    f"{top['dominant_core_signal'] or 'no dominant core signal'} = "
                    f"{_format_float(float(top.get('dominant_core_value', 0.0)))}"
                    + (
                        f", with diagnostics led by {top['dominant_extra_signal']} = "
                        f"{_format_float(float(top.get('dominant_extra_value', 0.0)))}."
                        if top.get("dominant_extra_signal") and float(top.get("dominant_extra_value", 0.0)) > 0.0
                        else "."
                    )
                ),
            }
        )
    else:
        findings.append(
            {
                "tone": "ok",
                "title": "Largest remaining gap",
                "body": (
                    "No remaining run-to-run gap was observed in the tested XPBD slice on this GPU/software stack."
                    if xpbd_only
                    else "No remaining run-to-run gap was observed in the tested harness slice on this GPU/software stack."
                ),
            }
        )

    observed_gaps: list[str] = []
    for result in sorted(
        drift,
        key=lambda item: (
            float(item.get("max_core_diff_overall", 0.0)),
            float(item.get("max_extras_diff_overall", 0.0)),
        ),
        reverse=True,
    ):
        scenario_title = SCENARIO_TITLES.get(result["scenario"], result["scenario"])
        note = (
            f"{scenario_title} / {result['solver']}: {result['unique_hashes']} unique hashes across {runs} runs; "
            f"dominant core drift in {result['dominant_core_signal'] or 'n/a'} = "
            f"{_format_float(float(result.get('dominant_core_value', 0.0)))}"
        )
        if result.get("dominant_extra_signal") and float(result.get("dominant_extra_value", 0.0)) > 0.0:
            note += (
                f"; dominant diagnostic drift in {result['dominant_extra_signal']} = "
                f"{_format_float(float(result.get('dominant_extra_value', 0.0)))}"
            )
        note += f"; execution = {'no graph' if result.get('disable_graph') else 'CUDA graph'}."
        observed_gaps.append(note)
    for result in failed:
        scenario_title = SCENARIO_TITLES.get(result["scenario"], result["scenario"])
        error_tail = (result.get("error") or "").splitlines()
        suffix = f" Last log line: {error_tail[-1].strip()}" if error_tail else ""
        observed_gaps.append(f"{scenario_title} / {result['solver']}: subprocess failure prevented analysis.{suffix}")
    if not observed_gaps:
        observed_gaps.append(
            "No observed run-to-run determinism gaps remain in this tested XPBD report slice."
            if xpbd_only
            else "No observed run-to-run determinism gaps remain in this tested report slice."
        )

    scope_limits = [
        "This report only checks repeated runs on the same machine, driver, GPU, and Warp/Newton build; it does not prove cross-GPU or cross-driver reproducibility.",
        "The verdict is based on final core state hashes. Videos and scenario extras are diagnostic evidence, not the primary pass/fail signal.",
    ]
    if collision_variant:
        scope_limits.append(
            "Rows labeled 'CollisionPipeline deterministic=True; collide Warp off' still run the rest of XPBD under the report's global Warp deterministic mode; only the custom collision pipeline is compiled and warmed under wp.config.deterministic='not_guaranteed'."
        )
    if any(result.get("disable_graph") for result in solver_results):
        scope_limits.append(
            "Rows marked no graph were replayed outside the CUDA graph path, so their verdicts do not cover graph-capture behavior."
        )

    return {
        "findings": findings,
        "observed_gaps": observed_gaps,
        "scope_limits": scope_limits,
    }


def _scenario_metrics(scenario: str, snapshot: Any) -> list[dict[str, str]]:
    extras = snapshot.extras

    if scenario == "falling_cube":
        heights = np.asarray(extras["final_height"], dtype=np.float64)
        return [
            _metric("Final height mean", float(heights.mean()), "m"),
            _metric(
                "Final height range", f"{_format_float(float(heights.min()))}-{_format_float(float(heights.max()))} m"
            ),
            _metric("Awake bodies", extras["awake_count"]),
            _metric("Mean speed", extras["mean_speed"], "m/s"),
        ]

    if scenario == "box_stack":
        contact_hist = np.asarray(extras["contact_count_history"])
        return [
            _metric("COM drift", extras["com_drift_world0"], "m"),
            _metric("Kinetic energy proxy", extras["kinetic_energy"]),
            _metric("Contact samples", contact_hist.size),
            _metric("Peak contacts", int(contact_hist.max()) if contact_hist.size else 0),
        ]

    if scenario == "domino_chain":
        prop = int(extras["propagation_time_steps"])
        prop_text = "not reached" if prop < 0 else f"{prop} steps"
        return [
            _metric("Fallen dominos", extras["fallen_count_world0"]),
            _metric("Last-domino time", prop_text),
            _metric("First angle", extras["first_domino_angle_deg"], "deg"),
            _metric("Tracked fall events", len(extras["fallen_step_map"])),
        ]

    if scenario == "arm_7dof":
        err = np.asarray(extras["final_target_error"], dtype=np.float64)
        tau = np.asarray(extras["joint_tau_history"], dtype=np.float64)
        contacts = np.asarray(extras["contact_count_history"])
        return [
            _metric("Max target error", float(err.max()), "rad"),
            _metric("Mean target error", float(err.mean()), "rad"),
            _metric("Peak PD torque", float(np.max(np.abs(tau))) if tau.size else 0.0, "N*m"),
            _metric("Peak contacts", int(contacts.max()) if contacts.size else 0),
        ]

    if scenario == "humanoid":
        root = np.asarray(extras["root_height_final"], dtype=np.float64)
        fell = np.asarray(extras["fell_mask"], dtype=np.uint8)
        com = np.asarray(extras["com_history"], dtype=np.float64)
        contacts = np.asarray(extras["foot_contact_history"])
        drift = 0.0
        if com.shape[0] >= 2:
            drift = float(np.linalg.norm(com[-1, :, :2] - com[0, :, :2], axis=1).max())
        return [
            _metric("Final root height", float(root.mean()), "m"),
            _metric("Fell worlds", int(fell.sum())),
            _metric("Max COM XY drift", drift, "m"),
            _metric("Peak contacts", int(contacts.max()) if contacts.size else 0),
        ]

    if scenario in {"vbd_cloth_patch", "vbd_soft_body"}:
        heights = np.asarray(extras["height_range"], dtype=np.float64)
        metrics = [
            _metric("Particles", extras["particle_count"]),
            _metric("Pinned particles", extras["pinned_particle_count"]),
            _metric(
                "Height range",
                f"{_format_float(float(heights[0]))}-{_format_float(float(heights[1]))} m",
            ),
        ]
        if "tet_count" in extras:
            metrics.append(_metric("Tets", extras["tet_count"]))
        return metrics

    return []


def _run_snapshot_subprocess(args: argparse.Namespace) -> int:
    from scripts import run_determinism

    run_determinism._apply_warp_deterministic(args.warp_deterministic)

    import newton

    scenario_args = argparse.Namespace(
        scenario=args.scenario,
        solver=args.solver,
        world_count=args.world_count,
        num_steps=args.num_steps,
        substeps=args.substeps,
        fps=args.fps,
        seed=args.seed,
        viewer="null",
        warp_deterministic=args.warp_deterministic,
        collision_pipeline_deterministic=args.collision_pipeline_deterministic,
        collision_pipeline_warp_deterministic=args.collision_pipeline_warp_deterministic,
    )
    scenario = run_determinism._build_scenario(scenario_args)
    if args.disable_graph:
        scenario._maybe_capture_graph = lambda: None

    viewer = newton.viewer.ViewerNull(num_frames=args.num_steps)
    scenario.build(viewer)
    for _ in range(args.num_steps):
        scenario.step()

    _write_snapshot(Path(args.snapshot_out), scenario.snapshot())
    return 0


def _compare_pair(
    scenario: str,
    solver: str,
    config: RunConfig,
    disable_graph: bool,
    variant: ReportVariant = DEFAULT_REPORT_VARIANT,
) -> dict[str, Any]:
    start = time.perf_counter()
    snapshots: list[Any] = []
    hashes: list[str] = []

    with tempfile.TemporaryDirectory(prefix="newton-det-report-") as tmp_name:
        tmp = Path(tmp_name)
        for run_index in range(config.runs):
            snapshot_path = tmp / f"run_{run_index:03d}.pkl"
            cmd = [
                sys.executable,
                str(THIS_FILE),
                "_snapshot-subrun",
                "--snapshot-out",
                str(snapshot_path),
                "--scenario",
                scenario,
                "--solver",
                solver,
                "--world-count",
                str(config.world_count),
                "--num-steps",
                str(config.num_steps),
                "--substeps",
                str(config.substeps),
                "--fps",
                str(config.fps),
                "--seed",
                str(config.seed),
                "--warp-deterministic",
                config.warp_deterministic,
            ]
            if variant.collision_pipeline_deterministic:
                cmd.append("--collision-pipeline-deterministic")
            if variant.collision_pipeline_warp_deterministic:
                cmd += [
                    "--collision-pipeline-warp-deterministic",
                    variant.collision_pipeline_warp_deterministic,
                ]
            if disable_graph:
                cmd.append("--disable-graph")

            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, check=False)
            if proc.returncode != 0:
                result = {
                    "scenario": scenario,
                    "solver": solver,
                    "variant_id": variant.id,
                    "variant_label": variant.label,
                    "variant_order": variant.sort_order,
                    "collision_pipeline_deterministic": variant.collision_pipeline_deterministic,
                    "collision_pipeline_wp_deterministic": variant.collision_pipeline_warp_deterministic,
                    "status": "failed",
                    "all_equal": False,
                    "disable_graph": disable_graph,
                    "duration_s": time.perf_counter() - start,
                    "error": "\n".join(
                        proc.stderr.decode(errors="replace").splitlines()[-24:]
                        + proc.stdout.decode(errors="replace").splitlines()[-8:]
                    ),
                }
                result["summary"] = _result_summary(result)
                return result
            snapshots.append(pickle.loads(snapshot_path.read_bytes()))
            hashes.append(_hash_core(snapshots[-1]))

    all_equal = len(set(hashes)) == 1
    first = snapshots[0]
    max_diff = _max_core_diff(snapshots)
    extras_diff = _extras_diff(snapshots)
    max_core = max(max_diff.values()) if max_diff else 0.0
    max_extras = max(extras_diff.values()) if extras_diff else 0.0
    dominant_core_signal, dominant_core_value = _dominant_diff(max_diff)
    dominant_extra_signal, dominant_extra_value = _dominant_diff(extras_diff)

    result = {
        "scenario": scenario,
        "solver": solver,
        "variant_id": variant.id,
        "variant_label": variant.label,
        "variant_order": variant.sort_order,
        "status": "bit_exact" if all_equal else "non_deterministic",
        "all_equal": all_equal,
        "hashes": hashes,
        "unique_hashes": len(set(hashes)),
        "max_core_diff": max_diff,
        "max_core_diff_overall": max_core,
        "dominant_core_signal": dominant_core_signal,
        "dominant_core_value": dominant_core_value,
        "extras_diff": extras_diff,
        "max_extras_diff_overall": max_extras,
        "dominant_extra_signal": dominant_extra_signal,
        "dominant_extra_value": dominant_extra_value,
        "disable_graph": disable_graph,
        "collision_pipeline_deterministic": bool(first.meta.get("collision_pipeline_deterministic")),
        "collision_pipeline_wp_deterministic": first.meta.get("collision_pipeline_wp_deterministic"),
        "graph_capture_enabled": bool(first.meta.get("graph_capture_enabled")),
        "meta": first.meta,
        "metrics": _scenario_metrics(scenario, first),
        "duration_s": time.perf_counter() - start,
    }
    result["summary"] = _result_summary(result)
    return result


def _capture_video(
    scenario: str,
    solver: str,
    config: RunConfig,
    out_dir: Path,
    frame_stride: int,
    width: int,
    height: int,
    video_fps: int,
    video_warp_deterministic: str,
) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "imageio and imageio-ffmpeg are required for MP4 output. "
            "Run with: uv run --with imageio --with imageio-ffmpeg python scripts/generate_determinism_report.py"
        ) from exc

    from scripts import run_determinism

    run_determinism._apply_warp_deterministic(video_warp_deterministic)

    import warp as wp

    import newton

    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    video_path = assets_dir / f"{scenario}_{solver}.mp4"
    poster_path = assets_dir / f"{scenario}_{solver}_poster.png"

    scenario_args = argparse.Namespace(
        scenario=scenario,
        solver=solver,
        world_count=1,
        num_steps=config.num_steps,
        substeps=config.substeps,
        fps=config.fps,
        seed=config.seed,
        viewer="gl",
        warp_deterministic=video_warp_deterministic,
        collision_pipeline_deterministic=False,
        collision_pipeline_warp_deterministic=None,
    )
    scene = run_determinism._build_scenario(scenario_args)
    scene._maybe_capture_graph = lambda: None
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True, vsync=False)
    scene.build(viewer)
    if scene.model is not None and scene.model.particle_count:
        viewer.show_particles = True
        viewer.show_springs = True

    pos, pitch, yaw, fov = CAMERAS[scenario]
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)
    viewer.camera.fov = fov

    frame_buffer = None
    captured = 0
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None

    writer = imageio.get_writer(
        video_path,
        fps=video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame in range(config.num_steps):
            scene.step()
            scene.render()
            if frame % frame_stride != 0 and frame != config.num_steps - 1:
                continue
            frame_buffer = viewer.get_frame(target_image=frame_buffer)
            arr = frame_buffer.numpy()
            if first_frame is None:
                first_frame = arr.copy()
            last_frame = arr.copy()
            writer.append_data(arr)
            captured += 1
    finally:
        writer.close()
        viewer.close()

    if first_frame is None or last_frame is None:
        raise RuntimeError(f"No frames captured for {scenario}/{solver}")
    Image.fromarray(first_frame).save(poster_path)

    return {
        "scenario": scenario,
        "solver": solver,
        "video": video_path.name,
        "poster": poster_path.name,
        "frames": captured,
        "sim_steps": config.num_steps,
        "frame_stride": frame_stride,
        "fps": video_fps,
        "warp_deterministic": video_warp_deterministic,
        "width": width,
        "height": height,
        "first_mean": float(first_frame.mean()) if first_frame is not None else 0.0,
        "last_mean": float(last_frame.mean()),
    }


def _capture_video_subprocess(
    scenario: str,
    solver: str,
    config: RunConfig,
    out_dir: Path,
    frame_stride: int,
    width: int,
    height: int,
    video_fps: int,
    video_warp_deterministic: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="newton-det-video-") as tmp_name:
        metadata_path = Path(tmp_name) / "video.json"
        cmd = [
            sys.executable,
            str(THIS_FILE),
            "_video-subrun",
            "--metadata-out",
            str(metadata_path),
            "--out-dir",
            str(out_dir),
            "--scenario",
            scenario,
            "--solver",
            solver,
            "--num-steps",
            str(config.num_steps),
            "--substeps",
            str(config.substeps),
            "--fps",
            str(config.fps),
            "--seed",
            str(config.seed),
            "--warp-deterministic",
            video_warp_deterministic,
            "--video-stride",
            str(frame_stride),
            "--video-width",
            str(width),
            "--video-height",
            str(height),
            "--video-fps",
            str(video_fps),
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, check=False)
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace").splitlines()[-24:]
            stdout_tail = proc.stdout.decode(errors="replace").splitlines()[-8:]
            raise RuntimeError(f"Video capture failed for {scenario}/{solver}:\n" + "\n".join(tail + stdout_tail))
        return json.loads(metadata_path.read_text(encoding="utf-8"))


def _run_video_subprocess(args: argparse.Namespace) -> int:
    config = RunConfig(
        runs=1,
        num_steps=args.num_steps,
        substeps=args.substeps,
        fps=args.fps,
        world_count=1,
        seed=args.seed,
        warp_deterministic=args.warp_deterministic,
    )
    metadata = _capture_video(
        args.scenario,
        args.solver,
        config,
        Path(args.out_dir),
        frame_stride=args.video_stride,
        width=args.video_width,
        height=args.video_height,
        video_fps=args.video_fps,
        video_warp_deterministic=args.warp_deterministic,
    )
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


def _status_label(result: dict[str, Any]) -> str:
    if result["status"] == "bit_exact":
        return "BIT-EXACT"
    if result["status"] == "non_deterministic":
        return "DRIFT"
    return "FAILED"


def _status_class(result: dict[str, Any]) -> str:
    if result["status"] == "bit_exact":
        return "ok"
    if result["status"] == "non_deterministic":
        return "warn"
    return "bad"


def _render_metric_grid(metrics: list[dict[str, str]]) -> str:
    return "\n".join(
        f"""
        <div class="metric">
          <span>{html.escape(metric["label"])}</span>
          <strong>{html.escape(metric["value"])}</strong>
        </div>"""
        for metric in metrics
    )


def _render_text_list(items: list[str]) -> str:
    return "\n".join(f"<li>{html.escape(item)}</li>" for item in items)


def _render_finding_cards(findings: list[dict[str, str]]) -> str:
    return "\n".join(
        f"""
        <article class="finding {html.escape(item["tone"])}">
          <p class="eyebrow">Key finding</p>
          <h3>{html.escape(item["title"])}</h3>
          <p>{html.escape(item["body"])}</p>
        </article>"""
        for item in findings
    )


def _render_solver_card(result: dict[str, Any]) -> str:
    execution = "no graph" if result.get("disable_graph") else "CUDA graph"
    dominant_core = (
        f"{result['dominant_core_signal']} ({_format_float(float(result.get('dominant_core_value', 0.0)))})"
        if result.get("dominant_core_signal") is not None
        else "none"
    )
    dominant_extra = (
        f"{result['dominant_extra_signal']} ({_format_float(float(result.get('dominant_extra_value', 0.0)))})"
        if result.get("dominant_extra_signal") is not None and float(result.get("dominant_extra_value", 0.0)) > 0.0
        else "none"
    )
    error_html = ""
    if result["status"] == "failed" and result.get("error"):
        error_html = f'<pre class="error-log">{html.escape(result["error"])}</pre>'

    return f"""
    <article class="solver-card {_status_class(result)}">
      <div class="solver-card-head">
        <div>
          <h3><code>{html.escape(result["solver"])}</code></h3>
        </div>
        <span class="pill {_status_class(result)}">{_status_label(result)}</span>
      </div>
      <p class="solver-summary">{html.escape(result.get("summary") or _result_summary(result))}</p>
      <div class="solver-stats">
        <div><span>Execution</span><strong>{html.escape(execution)}</strong></div>
        <div><span>Unique hashes</span><strong>{html.escape(str(result.get("unique_hashes", "-")))}</strong></div>
        <div><span>Dominant core drift</span><strong>{html.escape(dominant_core)}</strong></div>
        <div><span>Dominant extra drift</span><strong>{html.escape(dominant_extra)}</strong></div>
        <div><span>Max core diff</span><strong>{html.escape(_format_float(float(result.get("max_core_diff_overall", 0.0))))}</strong></div>
        <div><span>Duration</span><strong>{html.escape(_format_float(float(result.get("duration_s", 0.0))))} s</strong></div>
      </div>
      {error_html}
    </article>"""


def _render_solver_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for result in sorted(
        results,
        key=lambda item: (
            item["scenario"],
            item["solver"],
        ),
    ):
        if result.get("status") == "failed":
            diff = "-"
            extras = "-"
            hashes = "-"
            dominant_core = "-"
        else:
            diff = _format_float(float(result["max_core_diff_overall"]))
            extras = _format_float(float(result["max_extras_diff_overall"]))
            hashes = str(result["unique_hashes"])
            dominant_core = (
                f"{result['dominant_core_signal']} ({_format_float(float(result.get('dominant_core_value', 0.0)))})"
                if result.get("dominant_core_signal") is not None
                else "-"
            )
        rows.append(
            f"""
            <tr>
              <td>{html.escape(SCENARIO_TITLES.get(result["scenario"], result["scenario"]))}</td>
              <td><code>{html.escape(result["solver"])}</code></td>
              <td><span class="pill {_status_class(result)}">{_status_label(result)}</span></td>
              <td>{hashes}</td>
              <td>{html.escape(dominant_core)}</td>
              <td>{diff}</td>
              <td>{extras}</td>
              <td>{_format_float(float(result.get("duration_s", 0.0)))} s</td>
            </tr>"""
        )
    return "\n".join(rows)


def _render_scenario_sections(data: dict[str, Any]) -> str:
    results_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for result in data["solver_results"]:
        results_by_scenario.setdefault(result["scenario"], []).append(result)

    videos_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for video in data["videos"]:
        videos_by_scenario.setdefault(video["scenario"], []).append(video)

    sections = []
    for scenario in data["scenario_order"]:
        scenario_meta = data["scenarios"][scenario]
        representative_solver = scenario_meta["representative_solver"]
        primary = next(
            (result for result in results_by_scenario.get(scenario, []) if result["solver"] == representative_solver),
            None,
        )
        metrics = primary.get("metrics", []) if primary else []
        scenario_copy = scenario_meta["note"] or scenario_meta["doc"]
        if scenario_meta["note"] and scenario_meta["doc"]:
            scenario_copy = f"{scenario_meta['note']} {scenario_meta['doc']}"
        solver_cards = "\n".join(
            _render_solver_card(result)
            for result in sorted(
                results_by_scenario.get(scenario, []),
                key=lambda item: item["solver"],
            )
        )

        video_html = ""
        videos = sorted(videos_by_scenario.get(scenario, []), key=lambda item: item["solver"])
        if videos:
            rendered_videos = []
            for video in videos:
                src = f"assets/{html.escape(video['video'])}"
                poster = f"assets/{html.escape(video['poster'])}"
                rendered_videos.append(
                    f"""
                    <div class="solver-video">
                      <p class="video-label"><code>{html.escape(video["solver"])}</code></p>
                      <video controls muted loop playsinline preload="metadata" poster="{poster}">
                        <source src="{src}" type="video/mp4">
                      </video>
                      <p class="video-caption">
                        {video["frames"]} frames encoded at {video["fps"]} fps
                        ({video["width"]}x{video["height"]}, stride {video["frame_stride"]}).
                      </p>
                    </div>"""
                )
            video_html = "\n".join(rendered_videos)
        else:
            video_html = """<div class="video-missing">Video capture failed</div>"""

        sections.append(
            f"""
            <article class="scenario">
              <div class="media">
                {video_html}
              </div>
              <div class="scenario-body">
                <div class="scenario-heading">
                  <p class="eyebrow">{html.escape(scenario)}</p>
                  <h2>{html.escape(SCENARIO_TITLES.get(scenario, scenario))}</h2>
                </div>
                <p>{html.escape(scenario_copy)}</p>
                <div class="metrics">
                  {_render_metric_grid(metrics)}
                </div>
                <div class="solver-cards">
                  {solver_cards}
                </div>
              </div>
            </article>"""
        )
    return "\n".join(sections)


def _render_html(data: dict[str, Any]) -> str:
    cfg = data["config"]
    analysis = data["analysis"]
    scenario_count = len(data["scenario_order"])
    total = len(data["solver_results"])
    bit_exact = sum(1 for result in data["solver_results"] if result["status"] == "bit_exact")
    drift = sum(1 for result in data["solver_results"] if result["status"] == "non_deterministic")
    failed = sum(1 for result in data["solver_results"] if result["status"] == "failed")
    generated = html.escape(data["generated_at"])
    det_label = _humanize_mode(cfg["warp_deterministic"])
    has_collision_variant = any(
        result.get("variant_id") != DEFAULT_REPORT_VARIANT.id for result in data["solver_results"]
    )
    scope_text = (
        "This report covers every scenario in the harness that currently supports XPBD."
        if data.get("xpbd_only")
        else "The full report covers the physics scenarios registered for this report."
    )
    compare_text = (
        ' It also includes an XPBD comparison where the collision pipeline uses deterministic contact sorting while its Warp kernels are compiled with <code>wp.config.deterministic = "not_guaranteed"</code>.'
        if has_collision_variant
        else " Contact scenarios use <code>CollisionPipeline(deterministic=True)</code> for contact ordering."
    )
    report_title = (
        f"XPBD {det_label} determinism report"
        if data.get("xpbd_only")
        else f"{det_label.capitalize()} physics determinism report"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Newton Determinism Report</title>
  <link rel="icon" href="data:,">
  <style>
    :root {{
      color-scheme: light;
      --ink: #1b1f27;
      --muted: #5c6675;
      --line: #d9dee7;
      --paper: #f7f5f0;
      --panel: #ffffff;
      --ok: #087f5b;
      --ok-bg: #dff7ed;
      --warn: #a15c00;
      --warn-bg: #fff2cc;
      --bad: #a33a3a;
      --bad-bg: #ffe2df;
      --blue: #2456a6;
      --teal: #0f766e;
      --shadow: 0 18px 50px rgba(37, 43, 56, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      padding: 46px clamp(18px, 4vw, 64px) 28px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(120deg, rgba(36, 86, 166, 0.13), transparent 34%),
        linear-gradient(300deg, rgba(15, 118, 110, 0.12), transparent 38%),
        #fffdf8;
    }}
    .hero {{
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
      gap: 32px;
      align-items: end;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--teal);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      max-width: 860px;
      font-size: clamp(2.45rem, 6vw, 5.2rem);
      line-height: 0.95;
    }}
    .lead {{
      margin: 20px 0 0;
      max-width: 780px;
      color: var(--muted);
      font-size: clamp(1rem, 2vw, 1.18rem);
      line-height: 1.55;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .summary-tile {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
      box-shadow: var(--shadow);
    }}
    .summary-tile.emphasis {{
      border-color: rgba(36, 86, 166, 0.32);
      background: linear-gradient(180deg, rgba(36, 86, 166, 0.06), rgba(255, 255, 255, 0.98));
    }}
    .summary-tile span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .summary-tile strong {{
      display: block;
      margin-top: 6px;
      font-size: 1.75rem;
      line-height: 1;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px clamp(18px, 4vw, 64px) 64px;
    }}
    .band {{
      margin: 0 0 34px;
      padding: 24px 0;
      border-bottom: 1px solid var(--line);
    }}
    .band h2, .section-title {{
      margin: 0 0 14px;
      font-size: clamp(1.6rem, 3vw, 2.2rem);
    }}
    .findings-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .finding {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .finding.ok {{ border-left: 5px solid var(--ok); }}
    .finding.warn {{ border-left: 5px solid var(--warn); }}
    .finding.bad {{ border-left: 5px solid var(--bad); }}
    .finding h3 {{
      margin: 0 0 8px;
      font-size: 1.12rem;
    }}
    .finding p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .method-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .method {{
      border-left: 4px solid var(--blue);
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .method strong {{ display: block; margin-bottom: 5px; }}
    .method p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .scenario {{
      display: grid;
      grid-template-columns: minmax(320px, 1.05fr) minmax(320px, 0.95fr);
      gap: 22px;
      align-items: stretch;
      margin: 0 0 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .media {{
      min-height: 320px;
      background: #151821;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }}
    .solver-video {{
      background: #151821;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .solver-video:last-child {{ border-bottom: 0; }}
    .video-label {{
      margin: 0;
      padding: 10px 14px 0;
      color: #f2f4fa;
      background: #151821;
      font-size: 0.86rem;
      font-weight: 800;
    }}
    video {{
      display: block;
      width: 100%;
      height: auto;
      min-height: 320px;
      object-fit: cover;
      background: #151821;
    }}
    .video-caption {{
      margin: 0;
      padding: 10px 14px 14px;
      color: #d7dbe7;
      background: #151821;
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    .scenario-body {{
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .scenario-heading h2 {{
      margin: 0;
      font-size: clamp(1.45rem, 2.7vw, 2rem);
    }}
    .scenario-body p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
      background: #fbfcfd;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 7px;
      font-size: 1.18rem;
      overflow-wrap: anywhere;
    }}
    .solver-cards {{
      display: grid;
      gap: 10px;
    }}
    .solver-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfd;
    }}
    .solver-card.ok {{ border-left: 5px solid var(--ok); }}
    .solver-card.warn {{ border-left: 5px solid var(--warn); }}
    .solver-card.bad {{ border-left: 5px solid var(--bad); }}
    .solver-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .solver-card-head h3 {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .solver-summary {{
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .solver-stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .solver-stats div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      min-height: 74px;
    }}
    .solver-stats span {{
      display: block;
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .solver-stats strong {{
      display: block;
      margin-top: 6px;
      font-size: 1rem;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .error-log {{
      margin: 14px 0 0;
      padding: 12px;
      background: #161923;
      color: #f4f7ff;
      border-radius: 8px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.83rem;
      line-height: 1.45;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.92em;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 78px;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .pill.ok {{ color: var(--ok); background: var(--ok-bg); }}
    .pill.warn {{ color: var(--warn); background: var(--warn-bg); }}
    .pill.bad {{ color: var(--bad); background: var(--bad-bg); }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 960px;
    }}
    th, td {{
      padding: 13px 14px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }}
    th {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      background: #fbfcfd;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .list-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 24px;
    }}
    .list-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .list-card h3 {{
      margin: 0 0 10px;
      font-size: 1.15rem;
    }}
    .list-card p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .list-card ul {{
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.55;
    }}
    footer {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 clamp(18px, 4vw, 64px) 42px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 980px) {{
      .hero, .scenario, .list-grid {{ grid-template-columns: 1fr; }}
      .findings-grid {{ grid-template-columns: 1fr; }}
      .method-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .solver-stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      header {{ padding-top: 34px; }}
      main {{ padding-top: 24px; }}
      .method-grid, .metrics, .summary-grid, .solver-stats {{ grid-template-columns: 1fr; }}
      .scenario-body {{ padding: 18px; }}
      .media, video {{ min-height: 230px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="hero">
      <div>
        <p class="eyebrow">Newton determinism harness</p>
        <h1>{html.escape(report_title)}</h1>
        <p class="lead">
          {scenario_count} Newton scenarios were replayed in independent subprocesses with
          <code>wp.config.deterministic = "{html.escape(cfg["warp_deterministic"])}"</code>.
          The videos are captured from headless <code>ViewerGL.get_frame()</code>; the verdicts
          come from byte hashes of final physics state arrays. {html.escape(scope_text)}{compare_text}
        </p>
      </div>
      <div class="summary-grid">
        <div class="summary-tile emphasis"><span>Scenarios</span><strong>{scenario_count}</strong></div>
        <div class="summary-tile"><span>Solver pairs</span><strong>{total}</strong></div>
        <div class="summary-tile"><span>Bit-exact</span><strong>{bit_exact}</strong></div>
        <div class="summary-tile"><span>Drift</span><strong>{drift}</strong></div>
        <div class="summary-tile"><span>Failures</span><strong>{failed}</strong></div>
      </div>
    </div>
  </header>

  <main>
    <section class="band">
      <h2>What This Run Shows</h2>
      <div class="findings-grid">
        {_render_finding_cards(analysis["findings"])}
      </div>
    </section>

    <section class="band">
      <h2>Method</h2>
      <div class="method-grid">
        <div class="method">
          <strong>Independent replays</strong>
          <p>{cfg["runs"]} subprocess runs per supported scenario/solver pair.</p>
        </div>
        <div class="method">
          <strong>Canonical hash</strong>
          <p>Hashes include each scenario's canonical core arrays, such as rigid state, particle state, losses, and gradients.</p>
        </div>
        <div class="method">
          <strong>Fixed workload</strong>
          <p>{cfg["num_steps"]} frames, {cfg["substeps"]} substeps, {cfg["fps"]} Hz, seed {cfg["seed"]}, world count {cfg["world_count"]}.</p>
        </div>
        <div class="method">
          <strong>Deterministic contacts</strong>
          <p>Contact scenarios use <code>CollisionPipeline(deterministic=True)</code> with the report's global Warp determinism mode.</p>
        </div>
        <div class="method">
          <strong>Visual capture</strong>
          <p>Solver videos use one world and MP4 frames read through <code>ViewerGL.get_frame()</code>.</p>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Scenario Videos And Statistics</h2>
      {_render_scenario_sections(data)}
    </section>

    <section class="band">
      <h2>Solver Matrix</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Solver</th>
              <th>Verdict</th>
              <th>Unique hashes</th>
              <th>Dominant core signal</th>
              <th>Max core diff</th>
              <th>Max extras diff</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {_render_solver_table(data["solver_results"])}
          </tbody>
        </table>
      </div>
    </section>

    <section class="band">
      <h2>Observed Gaps And Scope</h2>
      <div class="list-grid">
        <div class="list-card">
          <h3>Observed gaps in this run</h3>
          <p>These notes are generated from the actual solver results above rather than a static message.</p>
          <ul>
            {_render_text_list(analysis["observed_gaps"])}
          </ul>
        </div>
        <div class="list-card">
          <h3>What the report does and does not prove</h3>
          <p>The current harness is precise about repeated local replays, but it intentionally stops short of broader claims.</p>
          <ul>
            {_render_text_list(analysis["scope_limits"])}
          </ul>
        </div>
        <div class="list-card">
          <h3>Controls in this harness</h3>
          <ul>
            <li><code>wp.config.deterministic</code> is set before heavy Warp/Newton imports and propagated to subprocesses.</li>
            <li>Each scenario owns a seeded NumPy RNG, so pose jitter and control trajectories are reproducible.</li>
            <li>The comparison boundary is process-level: no Python or GPU state is reused between replays.</li>
            <li>Only final core state decides the verdict; scenario extras are diagnostic and reported separately.</li>
          </ul>
        </div>
      </div>
    </section>
  </main>

  <footer>
    Generated {generated}. Raw machine-readable data: <code>determinism_report.json</code>.
  </footer>
</body>
</html>
"""


def _strip_trailing_whitespace(text: str) -> str:
    """Remove trailing spaces from generated HTML lines."""
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.determinism import SCENARIOS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        runs=args.runs,
        num_steps=args.num_steps,
        substeps=args.substeps,
        fps=args.fps,
        world_count=args.world_count,
        seed=args.seed,
        warp_deterministic=args.warp_deterministic,
    )

    scenario_order = _ordered_scenarios(SCENARIOS, xpbd_only=bool(args.xpbd_only))
    solver_results = []
    for scenario in scenario_order:
        scen_cls = SCENARIOS[scenario]
        solvers = ("xpbd",) if args.xpbd_only else scen_cls.supported_solvers
        for solver in solvers:
            for variant in _report_variants():
                # Keep MuJoCo on the normal graph-captured path. The no-graph
                # path can exceed the static deterministic counter bound in
                # dynamic narrow-phase kernels even when the scenario is
                # bit-exact under the standard harness path.
                disable_graph = False
                print(
                    f"[determinism] {scenario}/{solver} [{variant.label}] ({'no graph' if disable_graph else 'graph'})",
                    flush=True,
                )
                solver_results.append(
                    _compare_pair(
                        scenario,
                        solver,
                        config,
                        disable_graph=disable_graph,
                        variant=variant,
                    )
                )

    videos = []
    video_steps = config.num_steps
    if args.video_seconds is not None:
        video_steps = max(1, math.ceil(float(args.video_seconds) * args.video_fps * args.video_stride))
    video_config = dataclasses.replace(config, num_steps=video_steps)
    video_warp_deterministic = args.video_warp_deterministic or config.warp_deterministic
    captured_videos: set[tuple[str, str]] = set()
    for result in solver_results:
        if result.get("status") == "failed":
            continue
        scenario = result["scenario"]
        solver = result["solver"]
        key = (scenario, solver)
        if scenario not in CAMERAS or key in captured_videos:
            continue
        captured_videos.add(key)
        print(f"[video] {scenario}/{solver}", flush=True)
        videos.append(
            _capture_video_subprocess(
                scenario,
                solver,
                video_config,
                out_dir,
                frame_stride=args.video_stride,
                width=args.video_width,
                height=args.video_height,
                video_fps=args.video_fps,
                video_warp_deterministic=video_warp_deterministic,
            )
        )

    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "config": dataclasses.asdict(config),
        "scenario_order": scenario_order,
        "scenarios": {
            scenario: {
                "title": SCENARIO_TITLES.get(scenario, scenario),
                "doc": _short_doc(SCENARIOS[scenario]),
                "note": SCENARIO_NOTES.get(scenario, ""),
                "representative_solver": _representative_solver(
                    scenario,
                    SCENARIOS[scenario],
                    xpbd_only=bool(args.xpbd_only),
                ),
            }
            for scenario in scenario_order
        },
        "solver_results": solver_results,
        "videos": videos,
        "xpbd_only": bool(args.xpbd_only),
    }
    data["analysis"] = _build_analysis(
        solver_results,
        runs=config.runs,
        xpbd_only=bool(args.xpbd_only),
        warp_deterministic=config.warp_deterministic,
    )

    json_path = out_dir / "determinism_report.json"
    json_path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(_strip_trailing_whitespace(_render_html(data)), encoding="utf-8")

    print(f"[done] {html_path}")
    print(f"[done] {json_path}")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ViewerGL videos and a static Newton determinism report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Report output directory.")
    parser.add_argument("--runs", type=int, default=3, help="Independent subprocess runs per solver pair.")
    parser.add_argument("--num-steps", type=int, default=180, help="Simulation frames for stats and videos.")
    parser.add_argument("--substeps", type=int, default=10, help="Physics substeps per frame.")
    parser.add_argument("--fps", type=int, default=60, help="Simulation frame rate.")
    parser.add_argument("--world-count", type=int, default=1, help="World count for determinism statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Scenario seed.")
    parser.add_argument(
        "--warp-deterministic",
        default="run_to_run",
        choices=["not_guaranteed", "run_to_run", "gpu_to_gpu"],
        help="Warp deterministic mode set before Newton/Warp kernels are compiled.",
    )
    parser.add_argument("--video-stride", type=int, default=3, help="Capture every Nth frame for MP4 output.")
    parser.add_argument("--video-width", type=int, default=960, help="Video width.")
    parser.add_argument("--video-height", type=int, default=540, help="Video height.")
    parser.add_argument("--video-fps", type=int, default=24, help="Encoded video frame rate.")
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=None,
        help="Encoded video duration. When set, video sim steps are derived from duration, stride, and video fps.",
    )
    parser.add_argument(
        "--video-warp-deterministic",
        default=None,
        choices=["not_guaranteed", "run_to_run", "gpu_to_gpu"],
        help="Warp deterministic mode used only for video capture. Defaults to --warp-deterministic.",
    )
    parser.add_argument("--xpbd-only", action="store_true", help="Only run XPBD-supported scenarios with XPBD.")

    subrun = parser.add_subparsers(dest="command")
    snap = subrun.add_parser("_snapshot-subrun", help=argparse.SUPPRESS)
    snap.add_argument("--snapshot-out", required=True)
    snap.add_argument("--scenario", required=True)
    snap.add_argument("--solver", required=True)
    snap.add_argument("--world-count", type=int, required=True)
    snap.add_argument("--num-steps", type=int, required=True)
    snap.add_argument("--substeps", type=int, required=True)
    snap.add_argument("--fps", type=int, required=True)
    snap.add_argument("--seed", type=int, required=True)
    snap.add_argument("--warp-deterministic", required=True)
    snap.add_argument("--collision-pipeline-deterministic", action="store_true")
    snap.add_argument("--collision-pipeline-warp-deterministic", default=None)
    snap.add_argument("--disable-graph", action="store_true")

    video = subrun.add_parser("_video-subrun", help=argparse.SUPPRESS)
    video.add_argument("--metadata-out", required=True)
    video.add_argument("--out-dir", required=True)
    video.add_argument("--scenario", required=True)
    video.add_argument("--solver", required=True)
    video.add_argument("--num-steps", type=int, required=True)
    video.add_argument("--substeps", type=int, required=True)
    video.add_argument("--fps", type=int, required=True)
    video.add_argument("--seed", type=int, required=True)
    video.add_argument("--warp-deterministic", required=True)
    video.add_argument("--video-stride", type=int, required=True)
    video.add_argument("--video-width", type=int, required=True)
    video.add_argument("--video-height", type=int, required=True)
    video.add_argument("--video-fps", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "_snapshot-subrun":
        return _run_snapshot_subprocess(args)
    if args.command == "_video-subrun":
        return _run_video_subprocess(args)

    from scripts import run_determinism

    run_determinism._apply_warp_deterministic(args.warp_deterministic)
    _build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
