"""Command-line entry points for the scene-only pipeline.

Stages: ``prepare-scenes`` → ``plan-scene-cameras`` → ``render``.
Each stage is also runnable as a plain script through the thin launchers in this
directory (``python prepare_scenes.py``), which call :func:`main` with the stage
name already filled in.

Third-party imports are deferred into :func:`main`, so ``--help`` and a single
stage only need that stage's dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("datacompose")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .config import load_config

    config = load_config(args.config)

    if args.command == "prepare-scenes":
        from .scene_extract import prepare_scenes

        resume = None if args.resume is None else bool(args.resume)
        result = prepare_scenes(
            config,
            blender_bin=args.blender_bin,
            workers=args.workers,
            resume=resume,
        )
        _print_summary("scene", result)
        return
    if args.command == "plan-scene-cameras":
        result = _plan_scene_cameras(config, args)
        _print_summary("scene_camera_jobs", result)
        return
    if args.command == "render":
        from .render_scenes import render_scenes

        result = render_scenes(
            config,
            blender_bin=args.blender_bin,
            workers=args.workers,
            allow_partial=args.allow_partial,
        )
        _print_summary("render", result)
        return
    if args.command == "clear-render-partial":
        from .render_scenes import clear_render_partial

        result = clear_render_partial(config, job_id=args.job_id)
        _print_summary("render_partial", result)
        return
    parser.error(f"Unsupported command: {args.command}")


def _plan_scene_cameras(config, args) -> dict[str, Any]:
    from .io_utils import dump_json_atomic, dump_jsonl_atomic
    from .scene_cameras import CameraSpec, build_scene_camera_jobs, load_scene_document

    m4 = config.section("m4")
    camera = m4.get("camera", {}) or {}
    spec = CameraSpec(
        focal_length=float(camera.get("focal_length", 50.0)),
        yaw_degrees=tuple(float(v) for v in camera.get("yaw_degrees", [0.0, 360.0])),
        pitch_degrees=tuple(float(v) for v in camera.get("pitch_degrees", [-18.0, 12.0])),
        eye_height=tuple(float(v) for v in camera.get("eye_height", [1.10, 1.80])),
        height_policy=str(camera.get("height_policy", "floor_standing")),
        camera_radius=float(camera.get("camera_radius", 0.20)),
        near_clearance=float(camera.get("near_clearance", 0.30)),
        max_depth=float(camera.get("max_depth", 45.0)),
        ray_grid=int(camera.get("ray_grid", 24)),
        search_ray_grid=int(camera.get("search_ray_grid", 12)),
        min_hit_fraction=float(camera.get("min_hit_fraction", 0.90)),
        near_window=float(camera.get("near_window", 1.00)),
        max_near_fraction=float(camera.get("max_near_fraction", 0.25)),
        max_single_node_fraction=float(camera.get("max_single_node_fraction", 0.90)),
        min_depth_spread_ratio=float(camera.get("min_depth_spread_ratio", 1.35)),
        attempts_per_camera=int(camera.get("attempts_per_camera", 32)),
        min_camera_spacing=float(camera.get("min_camera_spacing", 1.50)),
    )
    cameras_per_scene = int(
        args.cameras_per_scene
        if args.cameras_per_scene is not None
        else m4.get("cameras_per_scene", 32)
    )
    scene_path = Path(args.scene_json).resolve() if args.scene_json else config.path("scene_output")
    document = build_scene_camera_jobs(
        load_scene_document(str(scene_path)),
        spec=spec,
        seed=int(args.seed if args.seed is not None else config.section("project").get("seed", 0)),
        cameras_per_scene=cameras_per_scene,
        fixture_categories={str(value) for value in m4.get("fixture_categories", [])},
        base_scene_license=m4.get("base_scene_license", {}),
        license_policy_version=str(m4.get("license_policy_version", "unconfigured")),
    )
    max_jobs = m4.get("max_render_jobs")
    jobs = document["jobs"]
    if max_jobs is not None and int(max_jobs) > 0:
        jobs = jobs[: int(max_jobs)]
    jobs_path = Path(args.output).resolve() if args.output else config.path("render_jobs_output")
    dump_jsonl_atomic(jobs_path, jobs)
    document = {**document, "jobs": jobs}
    # Side files follow the configured paths when present, so a minimal config
    # does not fail; the summary also lands next to an explicit --output.
    try:
        summary_path = config.path("render_jobs_summary")
    except KeyError:
        summary_path = jobs_path.with_suffix(".summary.json")
    dump_json_atomic(summary_path, document)
    try:
        rejects_path = config.path("render_job_rejects")
    except KeyError:
        rejects_path = jobs_path.with_suffix(".rejects.jsonl")
    dump_jsonl_atomic(rejects_path, document["rejects"])
    if not jobs:
        raise RuntimeError(
            "scene camera planning produced zero jobs; inspect the rejects and the scene graph"
        )
    return {**document, "jobs": jobs, "stats": {**document["stats"], "jobs": len(jobs)}}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datacompose",
        description="Render existing scenes under controlled lighting (scene-only synthesis).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scenes = subparsers.add_parser(
        "prepare-scenes", help="Normalise FBX scenes into .blend + scene.json"
    )
    _common_arguments(scenes)
    scenes.add_argument("--blender-bin", default=None)
    scenes.add_argument("--workers", type=int, default=None)
    resume_group = scenes.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=None)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")

    plan = subparsers.add_parser(
        "plan-scene-cameras", help="Plan collision-free cameras inside each scene"
    )
    _common_arguments(plan)
    plan.add_argument("--scene-json", default=None, help="Defaults to paths.scene_output")
    plan.add_argument("--output", default=None, help="Defaults to paths.render_jobs_output")
    plan.add_argument("--cameras-per-scene", type=int, default=None)
    plan.add_argument("--seed", type=int, default=None)

    render = subparsers.add_parser(
        "render", help="Render TokenLight components for the planned cameras"
    )
    _common_arguments(render)
    render.add_argument("--blender-bin", default=None)
    render.add_argument("--workers", type=int, default=None)
    render.add_argument("--allow-partial", action="store_true")

    clear_partial = subparsers.add_parser(
        "clear-render-partial",
        help="Explicitly clear one failed render partial after reviewing failure.json",
    )
    _common_arguments(clear_partial)
    clear_partial.add_argument("--job-id", required=True)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )


def _print_summary(kind: str, document: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "status": "complete",
                "kind": kind,
                "stats": document.get("stats", {}),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
