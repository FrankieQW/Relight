#!/usr/bin/env python3
"""Materialize operator-specific Stage-2 configs and a deterministic smoke fixture."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


TASKS = ("ambient_scale", "global_diffuse", "add_light", "in_scene_light")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repo-root", required=True)
    materialize.add_argument("--project-base", required=True)
    materialize.add_argument("--tokenlight-base", required=True)
    materialize.add_argument("--runtime-dir", required=True)

    verify_reuse = subparsers.add_parser("verify-reused-annotation")
    verify_reuse.add_argument("--project-config", required=True)

    smoke = subparsers.add_parser("prepare-smoke")
    smoke.add_argument("--formal-config", required=True)
    smoke.add_argument("--runtime-dir", required=True)
    smoke.add_argument("--max-rows", type=int, required=True)
    smoke.add_argument("--max-schedule-samples", type=int, required=True)

    resume = subparsers.add_parser("smoke-resume")
    resume.add_argument("--phase-one-config", required=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--output", required=True)

    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--formal-config", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-smoke")
    verify.add_argument("--formal-config", required=True)
    verify.add_argument("--fixture-metadata", required=True)
    verify.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        materialize(args)
    elif args.command == "verify-reused-annotation":
        summary = verify_reused_annotation(args.project_config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "prepare-smoke":
        prepare_smoke(args)
    elif args.command == "smoke-resume":
        smoke_resume(args)
    elif args.command == "evaluation":
        evaluation(args)
    else:
        verify_smoke(args)


def verify_reused_annotation(project_config_path: str | Path) -> dict[str, Any]:
    """Verify that frozen M1-M3 artifacts can safely be reused by M4."""

    config_path = Path(project_config_path).expanduser().resolve()
    config = load_yaml(config_path)
    project = config.get("project")
    paths = config.get("paths")
    if not isinstance(project, dict) or not isinstance(paths, dict):
        raise ValueError("project config must contain project and paths mappings")

    root_value = Path(os.path.expandvars(str(project.get("root", ".")))).expanduser()
    project_root = (
        root_value.resolve()
        if root_value.is_absolute()
        else (config_path.parent / root_value).resolve()
    )

    def configured_path(key: str) -> Path:
        if key not in paths:
            raise KeyError(f"project config is missing paths.{key}")
        value = Path(os.path.expandvars(str(paths[key]))).expanduser()
        return value.resolve() if value.is_absolute() else (project_root / value).resolve()

    object_path = configured_path("object_output")
    scene_path = configured_path("scene_output")
    annotation_path = configured_path("annotation_output")
    documents: dict[str, dict[str, Any]] = {}
    for label, path in (
        ("object.json", object_path),
        ("scene.json", scene_path),
        ("annotation_construction.json", annotation_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot reuse M1-M3 because {label} does not exist: {path}; "
                "set REUSE_ANNOTATION=false"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{label} root must be a JSON object: {path}")
        documents[label] = value

    annotation = documents["annotation_construction.json"]
    object_digest = f"sha256:{sha256_file(object_path)}"
    scene_digest = f"sha256:{sha256_file(scene_path)}"
    if annotation.get("object_digest") != object_digest:
        raise ValueError(
            "cannot reuse M1-M3: annotation_construction.json references a stale "
            f"object.json (annotation={annotation.get('object_digest')!r}, "
            f"current={object_digest}); set REUSE_ANNOTATION=false"
        )
    if annotation.get("scene_digest") != scene_digest:
        raise ValueError(
            "cannot reuse M1-M3: annotation_construction.json references a stale "
            f"scene.json (annotation={annotation.get('scene_digest')!r}, "
            f"current={scene_digest}); set REUSE_ANNOTATION=false"
        )

    scenes = documents["scene.json"].get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"scene.json contains no reusable scenes: {scene_path}")
    missing_blends: list[str] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict) or not scene.get("normalized_blend"):
            raise ValueError(f"scene.json scenes[{index}] has no normalized_blend")
        blend_value = Path(os.path.expandvars(str(scene["normalized_blend"]))).expanduser()
        blend_path = (
            blend_value.resolve()
            if blend_value.is_absolute()
            else (project_root / blend_value).resolve()
        )
        if not blend_path.is_file():
            missing_blends.append(str(blend_path))
    if missing_blends:
        preview = ", ".join(missing_blends[:3])
        suffix = "" if len(missing_blends) <= 3 else f" (+{len(missing_blends) - 3} more)"
        raise FileNotFoundError(
            f"cannot reuse M1-M3: {len(missing_blends)} normalized blend(s) are missing: "
            f"{preview}{suffix}; set REUSE_ANNOTATION=false"
        )

    return {
        "status": "reusable",
        "project_root": str(project_root),
        "object": str(object_path),
        "object_digest": object_digest,
        "scene": str(scene_path),
        "scene_digest": scene_digest,
        "annotation": str(annotation_path),
        "normalized_blends": len(scenes),
    }


def materialize(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    project = load_yaml(args.project_base)
    tokenlight = load_yaml(args.tokenlight_base)

    project.setdefault("project", {})["root"] = str(repo_root)
    allowlist = csv_strings(required_env("OBJECT_LICENSE_ALLOWLIST"))
    if not allowlist:
        raise ValueError("OBJECT_LICENSE_ALLOWLIST must contain at least one license")
    project["object"]["license_allowlist"] = allowlist
    project["scene"]["blender_bin"] = required_env("BLENDER_BIN")
    project["m4"]["blender_bin"] = required_env("BLENDER_BIN")
    project["m4"]["render_gpu_ids"] = csv_ints(required_env("RENDER_GPU_IDS"))
    project["m4"]["render_workers"] = int(required_env("RENDER_WORKERS"))
    project["m4"]["max_render_jobs"] = optional_int(os.environ.get("MAX_RENDER_JOBS"))
    project["m4"]["overwrite"] = False
    project["m4"]["license_policy_version"] = required_env("LICENSE_POLICY_VERSION")
    license_decision = required_env("BASE_SCENE_LICENSE_DECISION")
    if license_decision != "allowed":
        raise ValueError("formal Stage-2 data requires BASE_SCENE_LICENSE_DECISION=allowed")
    project["m4"]["base_scene_license"] = {
        "name": required_env("BASE_SCENE_LICENSE_NAME"),
        "source_uri": required_env("BASE_SCENE_SOURCE_URI"),
        "attribution": required_env("BASE_SCENE_ATTRIBUTION"),
        "decision": license_decision,
    }
    project["m4"]["render"]["resolution"] = int(required_env("RENDER_RESOLUTION"))
    project["m4"]["render"]["samples"] = int(required_env("RENDER_SAMPLES"))
    project["m4"]["render"]["persistent_data"] = env_bool("RENDER_PERSISTENT_DATA")
    project["m4"]["render"]["hdri_root"] = str(
        resolve_from_repo(repo_root, required_env("HDRI_ROOT"))
    )
    dataset_root = resolve_from_repo(repo_root, required_env("DATASET_ROOT"))
    project["paths"]["tokenlight_output"] = str(dataset_root)

    manifests = dataset_root / "manifests"
    tokenlight["paths"].update(
        {
            "upstream_checkpoint": required_env("STAGE1_CHECKPOINT"),
            "vae": required_env("VAE_PATH"),
            "dataset_root": str(dataset_root),
            "render_output_root": str(dataset_root),
            "render_jobs_manifest": str(
                resolve_from_repo(repo_root, str(project["paths"]["render_jobs_output"]))
            ),
            "train_manifest": str(manifests / "train.jsonl"),
            "validation_manifest": str(manifests / "validation.jsonl"),
            "test_manifest": str(manifests / "test.jsonl"),
            "output_root": str(resolve_from_repo(repo_root, required_env("TOKENLIGHT_OUTPUT_ROOT"))),
            "resume_checkpoint": optional_path("FORMAL_RESUME_CHECKPOINT"),
            "inference_checkpoint": None,
        }
    )
    tokenlight["data"]["resolution"] = int(required_env("RENDER_RESOLUTION"))
    tokenlight["data"]["num_workers"] = int(required_env("FORMAL_NUM_WORKERS"))
    tokenlight["data"]["require_composition_contract"] = True
    tokenlight["data"]["tasks"] = list(TASKS)
    tokenlight["model"]["fixture_mask_enabled"] = True
    tokenlight["train"].update(
        {
            "micro_batch_size": int(required_env("FORMAL_MICRO_BATCH_SIZE")),
            "gradient_accumulation_steps": int(required_env("FORMAL_GRADIENT_ACCUMULATION_STEPS")),
            "max_steps": int(required_env("FORMAL_MAX_STEPS")),
            "learning_rate": float(required_env("FORMAL_LEARNING_RATE")),
        }
    )
    tokenlight["train"]["smoke"]["enabled"] = False
    tokenlight["logging"]["run_id"] = required_env("FORMAL_RUN_ID")
    tokenlight["runtime"]["gpu_ids"] = csv_ints(required_env("TRAIN_GPU_IDS"))
    tokenlight["runtime"]["model_parallel_size"] = 1
    tokenlight["runtime"]["flash_attention"] = True
    tokenlight["runtime"]["activation_checkpointing"] = env_bool(
        "FORMAL_ACTIVATION_CHECKPOINTING"
    )

    dump_yaml(runtime_dir / "project.yaml", project)
    dump_yaml(runtime_dir / "tokenlight_stage2.yaml", tokenlight)
    print(json.dumps({"project": str(runtime_dir / "project.yaml"), "formal": str(runtime_dir / "tokenlight_stage2.yaml")}, ensure_ascii=False))


def prepare_smoke(args: argparse.Namespace) -> None:
    formal_path = Path(args.formal_config).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    formal = load_yaml(formal_path)
    source_manifest = Path(formal["paths"]["train_manifest"])
    rows = read_jsonl(source_manifest)
    selected = select_smoke_rows(rows, args.max_rows)
    smoke_root = runtime_dir / "smoke_dataset"
    manifest_root = smoke_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_root / "train.jsonl", selected)

    validation_rows = read_jsonl(Path(formal["paths"]["validation_manifest"]))
    if not validation_rows:
        raise ValueError("validation manifest is empty")
    write_jsonl(manifest_root / "validation.jsonl", validation_rows[: args.max_rows])
    release_path = Path(formal["paths"]["dataset_root"]) / "dataset_release.json"
    fixture_metadata = {
        "source_train_manifest": str(source_manifest.resolve()),
        "source_train_manifest_sha256": sha256_file(source_manifest),
        "source_validation_manifest_sha256": sha256_file(Path(formal["paths"]["validation_manifest"])),
        "dataset_release": str(release_path.resolve()),
        "dataset_release_sha256": sha256_file(release_path),
    }
    (smoke_root / "source.json").write_text(
        json.dumps(fixture_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    smoke = copy.deepcopy(formal)
    smoke["paths"].update(
        {
            "dataset_root": formal["paths"]["dataset_root"],
            "train_manifest": str(manifest_root / "train.jsonl"),
            "validation_manifest": str(manifest_root / "validation.jsonl"),
            "output_root": str(runtime_dir / "smoke_outputs"),
            "resume_checkpoint": None,
        }
    )
    smoke["data"]["num_workers"] = 0
    smoke["runtime"]["gpu_ids"] = [0]
    smoke["runtime"]["activation_checkpointing"] = True
    smoke["logging"]["run_id"] = required_env("SMOKE_RUN_ID")
    accumulation, schedule = find_covering_schedule(smoke, selected, args.max_schedule_samples)
    smoke["train"]["micro_batch_size"] = 1
    smoke["train"]["gradient_accumulation_steps"] = accumulation
    smoke["train"]["max_steps"] = 1
    smoke["train"]["checkpoint_every_steps"] = 1
    smoke["train"]["validation_every_steps"] = 1
    smoke["train"]["validation_batches"] = min(int(smoke["train"]["validation_batches"]), len(validation_rows[: args.max_rows]))
    smoke["train"]["smoke"].update(
        {"enabled": True, "required_tasks": list(TASKS), "summary_json": "smoke_summary.json"}
    )
    output = runtime_dir / "tokenlight_smoke_phase1.yaml"
    dump_yaml(output, smoke)
    print(json.dumps({"config": str(output), "accumulation_steps": accumulation, "schedule": schedule}, ensure_ascii=False))


def smoke_resume(args: argparse.Namespace) -> None:
    config = load_yaml(args.phase_one_config)
    config["paths"]["resume_checkpoint"] = str(Path(args.checkpoint).resolve())
    config["train"]["max_steps"] = int(config["train"]["max_steps"]) + 1
    dump_yaml(Path(args.output), config)


def evaluation(args: argparse.Namespace) -> None:
    config = load_yaml(args.formal_config)
    config["paths"]["inference_checkpoint"] = str(Path(args.checkpoint).resolve())
    config["paths"]["resume_checkpoint"] = None
    config["runtime"]["gpu_ids"] = [0]
    dump_yaml(Path(args.output), config)


def verify_smoke(args: argparse.Namespace) -> None:
    formal = load_yaml(args.formal_config)
    metadata = json.loads(Path(args.fixture_metadata).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("train_ready") is not True:
        raise RuntimeError("formal training refused: Stage-2 smoke is not train-ready")
    if summary.get("resume_verified") is not True:
        raise RuntimeError("formal training refused: Stage-2 smoke resume was not verified")
    checks = {
        "source_train_manifest_sha256": Path(formal["paths"]["train_manifest"]),
        "source_validation_manifest_sha256": Path(formal["paths"]["validation_manifest"]),
        "dataset_release_sha256": Path(formal["paths"]["dataset_root"]) / "dataset_release.json",
    }
    mismatches = [
        key for key, path in checks.items() if metadata.get(key) != sha256_file(path)
    ]
    if mismatches:
        raise RuntimeError(f"formal training refused: data changed after smoke: {mismatches}")
    print(json.dumps({"status": "pass", "verified": sorted(checks)}, ensure_ascii=False))


def select_smoke_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if max_rows < 1 or not rows:
        raise ValueError("smoke manifest requires at least one row")
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    for row in rows:
        supported = supported_tasks(row)
        if supported - covered:
            selected.append(row)
            covered.update(supported)
        if covered == set(TASKS):
            break
    missing = set(TASKS) - covered
    if missing:
        raise ValueError(f"full train manifest cannot cover smoke tasks: {sorted(missing)}")
    if len(selected) > max_rows:
        raise ValueError(
            f"SMOKE_MAX_MANIFEST_ROWS={max_rows} is too small to cover all tasks; need {len(selected)}"
        )
    selected_ids = {str(row["id"]) for row in selected}
    for row in rows:
        if len(selected) >= max_rows:
            break
        if str(row["id"]) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row["id"]))
    return selected


def supported_tasks(row: dict[str, Any]) -> set[str]:
    supported: set[str] = set()
    if row.get("ambient"):
        supported.add("ambient_scale")
    if len(row.get("diffuse", [])) >= 2:
        supported.add("global_diffuse")
    if row.get("point_lights"):
        supported.add("add_light")
    fixtures = row.get("in_scene_lights") or row.get("fixtures") or []
    if any(item.get("mask") and (item.get("path") or item.get("on")) for item in fixtures):
        supported.add("in_scene_light")
    return supported


def find_covering_schedule(
    config: dict[str, Any], rows: list[dict[str, Any]], limit: int
) -> tuple[int, list[str]]:
    import numpy as np

    # datacompose/stage2/config.py -> repository root is two levels up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Lumina-T2X" / "lumina_next_t2i"))
    from tokenlight.sampler_state import ResumableRandomSampler

    dataset_size = len(rows) * int(config["data"]["samples_per_scene_train"])
    sampler = ResumableRandomSampler(dataset_size, 0, limit, int(config["train"]["seed"]))
    schedule: list[str] = []
    seen: set[str] = set()
    probabilities = config["data"]["task_probabilities"]
    for dataset_index, sample_seed in sampler:
        row = rows[dataset_index % len(rows)]
        available = [task for task in TASKS if task in supported_tasks(row)]
        weights = np.asarray([float(probabilities.get(task, 1.0)) for task in available], dtype=np.float64)
        rng = np.random.default_rng(sample_seed)
        task = str(rng.choice(available, p=weights / weights.sum()))
        schedule.append(task)
        seen.add(task)
        if seen == set(TASKS):
            return len(schedule), schedule
    raise RuntimeError(f"no all-task smoke schedule found in {limit} samples; covered={sorted(seen)}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def dump_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable is empty: {name}")
    return value


def env_bool(name: str) -> bool:
    value = required_env(name).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    values = [int(item) for item in csv_strings(value)]
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise ValueError(f"invalid GPU list: {value}")
    return values


def optional_int(value: str | None) -> int | None:
    return int(value) if value and value.strip() else None


def optional_path(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return str(Path(value).expanduser().resolve()) if value else None


def resolve_from_repo(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    main()
