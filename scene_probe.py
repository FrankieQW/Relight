"""Stage ``probe-scenes``: bake triangles into clearance probes with Blender.

    python probe_scenes.py --blender-bin /opt/blender-4.5/blender

This stage is deliberately **separate** from both ``prepare-scenes`` and
``plan-scene-cameras``:

* it needs Blender (the planner does not), so planning stays a pure-NumPy step
  that can run anywhere;
* it is optional.  A scene that cannot be baked simply has no probe, and the
  planner falls back to the AABB free-space model for that scene, so an existing
  dataset can still be reproduced without ever running this stage.

Output is ``<scene_id>.probe.npz`` next to the normalised ``.blend``: the path is
derived from ``scene.json`` (see :func:`datacompose.probes.probe_path_for`)
rather than stored in it, so the scene graph stays a description of the scene
instead of a description of the pipeline's caches.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io_utils import dump_json_atomic, load_json, utc_now
from .probes import ProbeSpec, probe_path_for


LOGGER = logging.getLogger(__name__)


def probe_scenes(
    config: ProjectConfig,
    *,
    blender_bin: str | None = None,
    workers: int | None = None,
    resume: bool | None = None,
    scene_json: str | Path | None = None,
) -> dict[str, Any]:
    """Bake a probe for every scene that has a normalised ``.blend``."""
    m4 = config.section("m4")
    spec = ProbeSpec.from_mapping(m4.get("probe"))
    if not spec.enabled:
        LOGGER.info("m4.probe.enabled is false; nothing to bake")
        return {
            "stats": {"scenes": 0, "baked": 0, "skipped": 0, "failed": 0, "disabled": 1},
            "results": [],
            "failures": [],
        }
    blender = str(
        blender_bin
        or (m4.get("probe") or {}).get("blender_bin")
        or m4.get("blender_bin")
        or config.section("scene").get("blender_bin")
        or ""
    )
    if not blender:
        raise ValueError("m4.blender_bin (or m4.probe.blender_bin) is required")
    spec = replace(
        spec,
        workers=max(1, int(workers or spec.workers)),
        resume=bool(spec.resume if resume is None else resume),
    )
    script_path = Path(__file__).resolve().parent / "blender" / "probe_scene.py"
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    scene_path = Path(scene_json).resolve() if scene_json else config.path("scene_output")
    document = load_json(scene_path)
    work_root = config.path("scene_cache") / "probes"
    work_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scene in document.get("scenes", []) or []:
        scene_id = str(scene.get("scene_id"))
        blend = _resolve(config, str(scene.get("normalized_blend") or ""))
        if blend is None or not blend.is_file():
            failures.append(
                {
                    "scene_id": scene_id,
                    "reason": "missing_base_scene_blend",
                    "detail": str(blend),
                }
            )
            continue
        output = probe_path_for(blend)
        if spec.resume and _is_fresh(output, blend):
            skipped.append({"scene_id": scene_id, "output": str(output)})
            continue
        jobs.append(
            {
                "schema_version": "1.0",
                "scene_id": scene_id,
                "blend": str(blend),
                "output": str(output),
                "summary": str(work_root / f"{scene_id}.summary.json"),
                "log": str(work_root / f"{scene_id}.log"),
                "job": str(work_root / f"{scene_id}.job.json"),
                "resolution": float(spec.resolution),
                "max_voxels": int(spec.max_voxels),
                "clearance_cap": float(spec.clearance_cap),
                "padding": float(spec.padding),
            }
        )

    results: list[dict[str, Any]] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=spec.workers) as executor:
            futures = {
                executor.submit(_run_probe_job, job, blender, script_path): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001 - one report per stage.
                    LOGGER.warning(
                        "Probe bake failed for %s (planning will use AABBs): %s",
                        job["scene_id"],
                        error,
                    )
                    failures.append(
                        {
                            "scene_id": job["scene_id"],
                            "reason": "probe_failed",
                            "detail": str(error),
                        }
                    )

    summary = {
        "generated_at": utc_now(),
        "spec": spec.as_dict(),
        "stats": {
            "scenes": len(document.get("scenes", []) or []),
            "baked": len(results),
            "skipped": len(skipped),
            "failed": len(failures),
        },
        "results": sorted(results, key=lambda row: row["scene_id"]),
        "skipped": sorted(skipped, key=lambda row: row["scene_id"]),
        "failures": sorted(failures, key=lambda row: row["scene_id"]),
    }
    dump_json_atomic(work_root / "probe_summary.json", summary)
    if spec.require_all and failures:
        raise RuntimeError(
            f"{len(failures)} scene(s) could not be baked into a probe; "
            f"see {work_root / 'probe_summary.json'}"
        )
    return summary


def _resolve(config: ProjectConfig, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config.root / path).resolve()


def _is_fresh(output: Path, blend: Path) -> bool:
    """A probe is reusable when it exists and is not older than its blend."""
    if not output.is_file():
        return False
    try:
        return output.stat().st_mtime >= blend.stat().st_mtime
    except OSError:  # pragma: no cover - racy filesystem only
        return False


def _split_command(value: str) -> list[str]:
    """Split a launcher command into argv without eating Windows paths.

    ``shlex.split`` in its default POSIX mode treats a backslash as an escape
    character, so ``C:\\blender\\blender.exe`` comes back as
    ``C:blenderblender.exe``.  Quoted segments are honoured instead, which is
    what a wrapper command needs (``"C:\\Program Files\\...\\blender.exe"``).
    """
    text = str(value).strip()
    if not text:
        return []
    try:
        parts = shlex.split(text, posix=False)
    except ValueError:  # pragma: no cover - unbalanced quotes only
        return [text]
    # ``posix=False`` keeps the quotes as part of the token; strip them.
    return [
        part[1:-1]
        if len(part) > 1 and part[0] == part[-1] and part[0] in {'"', "'"}
        else part
        for part in parts
    ]


def _run_probe_job(job: dict[str, Any], blender: str, script: Path) -> dict[str, Any]:
    job_path = Path(job["job"])
    job_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_atomic(
        job_path,
        {key: value for key, value in job.items() if key not in {"job", "log"}},
    )
    output = Path(job["output"])
    if output.exists():
        # A stale probe must never be mistaken for a fresh one.
        output.unlink()
    log_path = Path(job["log"])
    # ``blender`` may be a wrapper command (``xvfb-run -a blender``), a versioned
    # path with spaces, or - in tests - an ordinary interpreter plus a script.
    # Splitting here keeps every caller from having to know about it.
    command = [
        *_split_command(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "probe",
        "--job",
        str(job_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"Blender exited with code {completed.returncode} for "
            f"{job['scene_id']}:\n{tail}"
        )
    if not output.is_file():
        raise RuntimeError(f"Blender did not produce a probe for {job['scene_id']}")
    summary_path = Path(job["summary"])
    if summary_path.is_file():
        return load_json(summary_path)
    return {"scene_id": job["scene_id"], "output": str(output)}


__all__ = ["probe_scenes"]
