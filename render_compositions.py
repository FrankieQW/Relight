from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .config import ProjectConfig
from .io_utils import dump_json_atomic, dump_jsonl_atomic, load_jsonl, utc_now
from .render_recovery import assert_no_existing_render_partials, clear_failed_render_partial


SUPPORTED_HDRI_SUFFIXES = {".hdr", ".exr"}


def clear_render_partial(config: ProjectConfig, *, job_id: str) -> dict[str, Any]:
    jobs_path = config.path("render_jobs_output")
    jobs = load_jsonl(jobs_path)
    if not jobs:
        raise ValueError(f"render job manifest is empty: {jobs_path}")
    _validated_job_ids(jobs, jobs_path)
    return clear_failed_render_partial(config.path("tokenlight_output"), jobs, job_id)


def render_compositions(
    config: ProjectConfig,
    *,
    blender_bin: str | None = None,
    workers: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    m4 = config.section("m4")
    jobs_path = config.path("render_jobs_output")
    jobs = load_jsonl(jobs_path)
    if not jobs:
        raise ValueError(f"render job manifest is empty: {jobs_path}")
    job_ids = _validated_job_ids(jobs, jobs_path)
    output_root = config.path("tokenlight_output")
    assert_no_existing_render_partials(output_root, jobs)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_blender = str(blender_bin or m4.get("blender_bin") or config.section("scene").get("blender_bin"))
    if not selected_blender:
        raise ValueError("m4.blender_bin or scene.blender_bin is required")
    object_root_env = str(m4.get("object_root_env", "OBJECT_ROOT"))
    object_root_value = os.environ.get(object_root_env)
    if not object_root_value:
        raise EnvironmentError(f"required environment variable is not set: {object_root_env}")
    object_root = Path(object_root_value).expanduser().resolve()
    if not object_root.is_dir():
        raise NotADirectoryError(object_root)

    gpu_ids = [int(value) for value in m4.get("render_gpu_ids", [0])]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(value < 0 for value in gpu_ids):
        raise ValueError("m4.render_gpu_ids must be unique non-negative integers")
    worker_count = max(1, int(workers or m4.get("render_workers", len(gpu_ids))))
    worker_count = min(worker_count, len(jobs))
    render_config = _resolved_render_config(config, m4.get("render", {}))
    runtime_root = output_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    script = config.root / "datacompose" / "blender" / "render_composition.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    processes: list[tuple[int, list[str], subprocess.Popen[str], Path, Any]] = []
    for worker_index in range(worker_count):
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        snapshot = runtime_root / f"composition_worker_{worker_index:03d}.json"
        worker_runtime = {
            "project_root": str(config.root),
            "object_root": str(object_root),
            "jobs_path": str(jobs_path),
            "output_root": str(output_root),
            "worker_index": worker_index,
            "worker_count": worker_count,
            "gpu_id": gpu_id,
            "render": render_config,
            "fixture": m4.get("fixture", {}),
            "overwrite": bool(m4.get("overwrite", False)),
            "allow_partial": bool(allow_partial),
        }
        dump_json_atomic(snapshot, worker_runtime)
        log_path = runtime_root / f"composition_worker_{worker_index:03d}.log"
        command = [
            selected_blender,
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--runtime-config",
            str(snapshot),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Write each worker directly to its own file.  PIPE would allow an
        # uncollected worker to fill its buffer while we wait on another one.
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        processes.append((worker_index, command, process, log_path, log_handle))

    worker_failures: list[dict[str, Any]] = []
    for worker_index, command, process, log_path, log_handle in processes:
        process.wait()
        log_handle.close()
        if process.returncode:
            worker_failures.append(
                {
                    "worker": worker_index,
                    "return_code": process.returncode,
                    "command": command,
                    "log": str(log_path),
                }
            )

    render_errors: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        render_errors.extend(
            load_jsonl(output_root / "render_workers" / f"worker_{worker_index:03d}" / "errors.jsonl")
        )
    dump_jsonl_atomic(output_root / "render_errors.jsonl", render_errors)
    expected_metadata = {
        job_id: output_root / "components" / job_id / "metadata.json" for job_id in job_ids
    }
    missing_job_outputs = [job_id for job_id, path in expected_metadata.items() if not path.is_file()]
    metadata_count = len(jobs) - len(missing_job_outputs)
    completed_ids = {
        path.parent.name for path in (output_root / "components").glob("*/metadata.json")
    }
    foreign_completed_ids = sorted(completed_ids - set(job_ids))
    summary = {
        "generated_at": utc_now(),
        "jobs": len(jobs),
        "rendered_or_reused": metadata_count,
        "render_errors": len(render_errors),
        "worker_failures": worker_failures,
        "missing_job_outputs": missing_job_outputs,
        "foreign_completed_outputs": foreign_completed_ids,
        "allow_partial": bool(allow_partial),
    }
    dump_json_atomic(output_root / "render_summary.json", summary)
    if worker_failures and not allow_partial:
        raise RuntimeError(f"{len(worker_failures)} Blender composition worker(s) failed")
    if render_errors and not allow_partial:
        raise RuntimeError(
            f"{len(render_errors)} render job(s) failed; see {output_root / 'render_errors.jsonl'}"
        )
    if missing_job_outputs and not allow_partial:
        raise RuntimeError(
            f"{len(missing_job_outputs)} render job(s) produced no metadata; "
            f"see {output_root / 'render_summary.json'}"
        )
    return summary


def _resolved_render_config(config: ProjectConfig, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("m4.render must be a mapping")
    result = dict(value)
    hdri_value = result.get("hdri_root")
    if hdri_value in (None, ""):
        return result
    hdri_root = Path(os.path.expandvars(str(hdri_value))).expanduser()
    if not hdri_root.is_absolute():
        hdri_root = config.root / hdri_root
    hdri_root = hdri_root.resolve()
    if not hdri_root.is_dir():
        raise NotADirectoryError(f"m4.render.hdri_root is not a directory: {hdri_root}")
    if not any(
        path.is_file() and path.suffix.lower() in SUPPORTED_HDRI_SUFFIXES
        for path in hdri_root.rglob("*")
    ):
        raise ValueError(f"m4.render.hdri_root contains no .hdr or .exr files: {hdri_root}")
    result["hdri_root"] = str(hdri_root)
    return result


def _validated_job_ids(jobs: list[dict[str, Any]], jobs_path: Path) -> list[str]:
    job_ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not job_id for job_id in job_ids) or len(set(job_ids)) != len(job_ids):
        raise ValueError(
            f"render job manifest contains missing or duplicate job_id values: {jobs_path}"
        )
    return job_ids
