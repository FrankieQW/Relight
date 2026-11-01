from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .config import load_config
from .geometry_prepare import prepare_geometry
from .llm_annotate import annotate_construction
from .object_index import prepare_objects
from .render_compositions import clear_render_partial, render_compositions
from .render_jobs import build_render_jobs
from .scene_extract import prepare_scenes


LOGGER = logging.getLogger("datacompose")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)

    if args.command == "prepare-objects":
        if args.inventory_mode != "markdown":
            parser.error("Only the confirmed 80-object markdown inventory mode is supported")
        result = prepare_objects(config)
        _print_summary("object", result)
        return
    if args.command == "prepare-scenes":
        resume = None if args.resume is None else bool(args.resume)
        result = prepare_scenes(
            config,
            blender_bin=args.blender_bin,
            workers=args.workers,
            resume=resume,
        )
        _print_summary("scene", result)
        return
    if args.command == "annotate-construction":
        result = asyncio.run(
            annotate_construction(
                config,
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                allow_partial=args.allow_partial,
            )
        )
        _print_summary("annotation", result)
        return
    if args.command == "prepare-geometry":
        result = prepare_geometry(config)
        _print_summary("prepared_geometry", result)
        return
    if args.command == "build-render-jobs":
        result = build_render_jobs(
            config,
            annotations_path=Path(args.annotations).resolve() if args.annotations else None,
            output_path=Path(args.output).resolve() if args.output else None,
            seed=args.seed,
        )
        _print_summary("render_jobs", result)
        return
    if args.command == "render":
        result = render_compositions(
            config,
            blender_bin=args.blender_bin,
            workers=args.workers,
            allow_partial=args.allow_partial,
        )
        _print_summary("render", result)
        return
    if args.command == "clear-render-partial":
        result = clear_render_partial(config, job_id=args.job_id)
        _print_summary("render_partial", result)
        return
    parser.error(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datacompose",
        description=(
            "Prepare Objaverse, FBX scene, construction annotation, render jobs and "
            "TokenLight components."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    objects = subparsers.add_parser("prepare-objects", help="Build data/object/object.json")
    _common_arguments(objects)
    objects.add_argument("--inventory-mode", choices=("markdown",), default="markdown")
    objects.add_argument("--workers", type=int, default=None, help="Reserved for future I/O sharding")

    scenes = subparsers.add_parser("prepare-scenes", help="Build normalized scene blends and scene.json")
    _common_arguments(scenes)
    scenes.add_argument("--blender-bin", default=None)
    scenes.add_argument("--workers", type=int, default=None)
    resume_group = scenes.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", dest="resume", default=None)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")

    annotate = subparsers.add_parser(
        "annotate-construction", help="Build data/annotation_construction.json with vLLM"
    )
    _common_arguments(annotate)
    annotate.add_argument("--base-url", default=None)
    annotate.add_argument("--model", default=None)
    annotate.add_argument("--concurrency", type=int, default=None)
    annotate.add_argument("--allow-partial", action="store_true", default=None)
    annotate.add_argument("--resume", action="store_true", help="Accepted for command compatibility; cache reuse is automatic")

    geometry = subparsers.add_parser(
        "prepare-geometry", help="Validate object scale/orientation contracts for M4"
    )
    _common_arguments(geometry)
    geometry.add_argument("--workers", type=int, default=None, help="Reserved for Blender geometry caching")

    jobs = subparsers.add_parser(
        "build-render-jobs", help="Build deterministic object-scene composition jobs"
    )
    _common_arguments(jobs)
    jobs.add_argument("--annotations", default=None)
    jobs.add_argument("--output", default=None)
    jobs.add_argument("--seed", type=int, default=None)

    render = subparsers.add_parser("render", help="Render TokenLight components from composition jobs")
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
