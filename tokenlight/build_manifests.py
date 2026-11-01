from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datacompose import coordinates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild UID-safe TokenLight manifests from rendered metadata.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    output_root = Path(config["paths"]["render_output_root"]).expanduser()
    split_profile = str(config["data"].get("split_profile", "object-held-out"))
    if split_profile not in {"object-held-out", "scene-held-out"}:
        raise ValueError("data.split_profile must be object-held-out or scene-held-out")
    require_composition = bool(config["data"].get("require_composition_contract", False))
    ignored_foreign_outputs: list[str] = []
    if require_composition:
        jobs_path_value = config["paths"].get("render_jobs_manifest")
        if not jobs_path_value:
            raise ValueError("paths.render_jobs_manifest is required for composition data")
        jobs_path = Path(jobs_path_value).expanduser()
        jobs = load_jsonl(jobs_path)
        if not jobs:
            raise FileNotFoundError(f"render job manifest is empty: {jobs_path}")
        jobs_by_id = {str(job.get("job_id") or ""): job for job in jobs}
        if "" in jobs_by_id or len(jobs_by_id) != len(jobs):
            raise ValueError(f"render job manifest contains missing or duplicate job IDs: {jobs_path}")
        rows = []
        for job_id, job in sorted(jobs_by_id.items()):
            metadata_path = output_root / "components" / job_id / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"render job has no completed metadata: {metadata_path}")
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
            _validate_job_metadata(row, job, metadata_path)
            rows.append(row)
        completed_ids = {
            path.parent.name for path in (output_root / "components").glob("*/metadata.json")
        }
        ignored_foreign_outputs = sorted(completed_ids - set(jobs_by_id))
    else:
        metadata_paths = sorted((output_root / "components").glob("*/metadata.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"没有找到 metadata.json: {output_root / 'components'}")
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        row = with_camera_coordinates(row)
        uid = str(row.get("asset_uid") or Path(row["asset"]).stem)
        row["asset_uid"] = uid
        if require_composition:
            if not row.get("base_scene_id") or not row.get("lineage"):
                raise ValueError(f"composition metadata incomplete: {row.get('id')}")
            if row.get("license", {}).get("decision") != "allowed":
                raise ValueError(f"license gate rejected sample: {row.get('id')}")
        group = uid if split_profile == "object-held-out" else str(row.get("base_scene_id", ""))
        if not group:
            raise ValueError(f"sample has no split group for {split_profile}: {row.get('id')}")
        grouped.setdefault(group, []).append(row)

    render = config["render"]
    fractions = [float(render[key]) for key in ("train_fraction", "validation_fraction", "test_fraction")]
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError("render split fraction 之和必须为 1")
    group_ids = sorted(grouped)
    if split_profile == "scene-held-out" and len(group_ids) < 3:
        raise ValueError("scene-held-out requires at least three distinct base scenes")
    random.Random(int(render["split_seed"])).shuffle(group_ids)
    validation_count = round(len(group_ids) * fractions[1])
    test_count = round(len(group_ids) * fractions[2])
    validation = set(group_ids[:validation_count])
    test = set(group_ids[validation_count : validation_count + test_count])
    split_rows = {"train": [], "validation": [], "test": []}
    for group, group_rows in grouped.items():
        split = "validation" if group in validation else "test" if group in test else "train"
        split_rows[split].extend(group_rows)
    for split, values in split_rows.items():
        write_jsonl(output_root / "manifests" / f"{split}.jsonl", sorted(values, key=lambda row: row["id"]))
    release = {
        "schema_version": 1,
        "split_profile": split_profile,
        "split_seed": int(render["split_seed"]),
        "counts": {split: len(values) for split, values in split_rows.items()},
        "groups": {split: len({group_key(row, split_profile) for row in values}) for split, values in split_rows.items()},
        "fixture_sources": fixture_source_counts(rows),
        "ignored_foreign_completed_outputs": ignored_foreign_outputs,
        "license_policy_versions": sorted(
            {str(row.get("license", {}).get("policy_version")) for row in rows}
        ),
        "lineage_digests": sorted(
            {
                value
                for row in rows
                for key, value in row.get("lineage", {}).items()
                if key.endswith("digest")
            }
        ),
        "manifest_sha256": {
            split: sha256_file(output_root / "manifests" / f"{split}.jsonl")
            for split in split_rows
        },
        "intrinsic_passes": intrinsic_passes_payload(rows, render),
        "camera_position": {
            "axes": coordinates.CAMERA_AXES,
            "depth": "signed_distance_along_camera_forward_axis",
            "rows_with_image_plane_pixels": sum(
                1 for values in split_rows.values() for row in values
                if has_image_plane_pixels(row)
            ),
        },
    }
    (output_root / "dataset_release.json").write_text(
        json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, ensure_ascii=False), flush=True)


def load_yaml(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config 根节点必须是 mapping")
    return config


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _validate_job_metadata(row: dict, job: dict, path: Path) -> None:
    expected_digest = str(job.get("lineage", {}).get("render_job_digest") or "")
    actual_digest = str(row.get("lineage", {}).get("render_job_digest") or "")
    if row.get("id") != job.get("job_id"):
        raise ValueError(f"metadata ID does not match render job: {path}")
    if row.get("asset_uid") != job.get("object_uid"):
        raise ValueError(f"metadata asset UID does not match render job: {path}")
    if row.get("base_scene_id") != job.get("base_scene_id"):
        raise ValueError(f"metadata base scene does not match render job: {path}")
    if not expected_digest or actual_digest != expected_digest:
        raise ValueError(f"metadata render-job lineage does not match current manifest: {path}")


def with_camera_coordinates(row: dict) -> dict:
    """Attach OpenCV camera coordinates and image-plane data to one manifest row.

    Training reads ``camera_position`` = ``[right, down, forward]`` in meters
    relative to the camera center, so every shipped manifest row must carry it.
    ``image_plane`` adds pixel ``u``/``v`` plus ``depth`` when the camera block
    has intrinsics; older metadata still gets normalized coordinates and depth.
    The canonical ``position`` field is preserved unchanged.
    """
    try:
        converted, _, _ = coordinates.convert_row(row, require_intrinsics=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot derive camera coordinates for {row.get('id')!r}: {error}") from error
    return converted


def has_image_plane_pixels(row: dict) -> bool:
    """True when any light in the row carries pixel ``u``/``v`` (intrinsics present)."""
    lights = list(row.get("point_lights") or []) + list(row.get("in_scene_lights") or [])
    for light in lights:
        plane = light.get("image_plane") if isinstance(light, dict) else None
        if isinstance(plane, dict) and "u" in plane:
            return True
    return False


def group_key(row: dict, split_profile: str) -> str:
    return str(row["asset_uid"] if split_profile == "object-held-out" else row["base_scene_id"])


def fixture_source_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"scene_native": 0, "procedural_fallback": 0}
    for row in rows:
        for fixture in row.get("in_scene_lights", []):
            source = fixture.get("fixture_source")
            if source in counts:
                counts[source] += 1
    return counts


def intrinsic_passes_payload(rows: list[dict], render: dict) -> dict:
    """Summarise intrinsic pass availability for ``dataset_release.json``.

    ``enabled``/``rows_with_passes`` are derived from the shipped metadata so the
    release stays truthful even if only some jobs were re-rendered with passes.
    """
    render_passes = (render or {}).get("passes") or {}
    definitions: dict = {}
    for row in rows:
        block = row.get("passes")
        if isinstance(block, dict) and isinstance(block.get("definition"), dict):
            definitions = dict(block["definition"])
            break
    return {
        "enabled": any(isinstance(row.get("passes"), dict) for row in rows),
        "rows_with_passes": sum(1 for row in rows if isinstance(row.get("passes"), dict)),
        "definitions": definitions,
        "quality_thresholds": {
            "min_base_color_linked_ratio": float(
                render_passes.get("min_base_color_linked_ratio", 0.9)
            ),
            "max_magenta_ratio": float(render_passes.get("max_magenta_ratio", 1e-4)),
            "min_textured_pixel_ratio": 0.02,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


if __name__ == "__main__":
    main()
