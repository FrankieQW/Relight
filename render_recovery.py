from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Iterable


def assert_no_existing_render_partials(
    output_root: Path,
    jobs: Iterable[dict[str, Any]],
) -> None:
    """Fail a batch before Blender starts if any manifest job has a partial."""
    conflicts: list[Path] = []
    for job in jobs:
        job_id = job.get("job_id")
        partial = _job_path(output_root, job_id, suffix=".partial")
        if partial.exists():
            failure = partial / "failure.json"
            conflicts.append(failure if failure.is_file() else partial)
    if conflicts:
        locations = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            "render batch refused because existing partial output must be reviewed and cleared "
            f"one job at a time; preserved evidence: {locations}"
        )


def prepare_render_partial(output_root: Path, job_id: str) -> Path:
    """Create a new partial directory without destroying prior diagnostics."""
    partial = _job_path(output_root, job_id, suffix=".partial")
    if partial.exists():
        failure = partial / "failure.json"
        evidence = failure if failure.is_file() else partial
        raise FileExistsError(
            f"render partial already exists for job {job_id!r}; preserved evidence: {evidence}. "
            "After reviewing the failure, clear only this job with "
            "`python -m datacompose.cli clear-render-partial "
            f"--config <config> --job-id {job_id}`."
        )
    partial.mkdir(parents=True)
    return partial


def clear_failed_render_partial(
    output_root: Path,
    jobs: Iterable[dict[str, Any]],
    job_id: str,
) -> dict[str, Any]:
    """Remove one manifest-bound failed partial so that it can be retried explicitly."""
    matches = [job for job in jobs if job.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError(
            f"job_id must identify exactly one render job in the manifest: {job_id!r} "
            f"(matches={len(matches)})"
        )

    final_directory = _job_path(output_root, job_id)
    metadata = final_directory / "metadata.json"
    if metadata.is_file():
        raise FileExistsError(f"completed render output already exists: {metadata}")

    partial = _job_path(output_root, job_id, suffix=".partial")
    failure = partial / "failure.json"
    if not partial.is_dir():
        raise FileNotFoundError(f"failed render partial does not exist: {partial}")
    if not failure.is_file():
        raise FileNotFoundError(
            "refusing to clear partial without failure evidence (it may still be active): "
            f"{failure}"
        )

    shutil.rmtree(partial)
    return {
        "stats": {"cleared": 1},
        "job_id": job_id,
        "cleared_partial": str(partial),
    }


def _job_path(output_root: Path, job_id: str, *, suffix: str = "") -> Path:
    if not isinstance(job_id, str) or not job_id or Path(job_id).name != job_id:
        raise ValueError(f"job_id must be a non-empty path-safe name: {job_id!r}")
    components = (Path(output_root) / "components").resolve()
    candidate = (components / f"{job_id}{suffix}").resolve()
    if candidate.parent != components:
        raise ValueError(f"job path escapes the components directory: {job_id!r}")
    return candidate
