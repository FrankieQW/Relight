"""Build scene-held-out TokenLight manifests from rendered components.

Reads ``<dataset_root>/components/*/metadata.json`` (written by
``blender/render_scene.py``), groups samples by ``scene_id`` and writes
``manifests/{train,validation,test}.jsonl`` plus ``dataset_release.json``.

Splitting is always scene-held-out: a scene must never appear in two splits,
because every camera in it shares geometry, materials and lighting context.

    python tokenlight/build_manifests.py --config stage2.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import yaml

# Run as a plain script (`python tokenlight/build_manifests.py --config x.yaml`)
# from any working directory and from a renamed project folder.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
from _launcher import ensure_package  # noqa: E402

ensure_package()

from datacompose import coordinates  # noqa: E402


REQUIRED_ROW_FIELDS = ("id", "scene_id", "camera", "canonical", "lineage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scene-held-out TokenLight manifests.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    output_root = Path(config["paths"]["render_output_root"]).expanduser()
    dataset_config = config.get("data", {})
    split_profile = str(dataset_config.get("split_profile", "scene-held-out"))
    if split_profile != "scene-held-out":
        raise ValueError(
            "scene-only data is split by scene; data.split_profile must be scene-held-out"
        )

    metadata_paths = sorted((output_root / "components").glob("*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no metadata.json under {output_root / 'components'}")

    rows: list[dict] = []
    for path in metadata_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        missing = [field for field in REQUIRED_ROW_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"metadata is not scene-only ({', '.join(missing)}): {path}")
        if not row["canonical"].get("position_axes") == coordinates.CANONICAL_AXES:
            raise ValueError(f"unsupported canonical axes in {path}")
        rows.append(with_camera_coordinates(row))

    render = config["render"]
    fractions = [float(render[key]) for key in ("train_fraction", "validation_fraction", "test_fraction")]
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError("render split fractions must sum to 1")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene_id"]), []).append(row)
    scene_ids = sorted(grouped)
    if len(scene_ids) < 3:
        raise ValueError("scene-held-out splitting requires at least three distinct scenes")

    random.Random(int(render["split_seed"])).shuffle(scene_ids)
    validation_count = round(len(scene_ids) * fractions[1])
    test_count = round(len(scene_ids) * fractions[2])
    validation = set(scene_ids[:validation_count])
    test = set(scene_ids[validation_count : validation_count + test_count])
    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for scene_id, scene_rows in grouped.items():
        split = "validation" if scene_id in validation else "test" if scene_id in test else "train"
        split_rows[split].extend(scene_rows)

    for split, values in split_rows.items():
        write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            sorted(values, key=lambda row: row["id"]),
        )

    release = {
        "schema_version": 2,
        "contract": "scene_only_v1",
        "split_profile": split_profile,
        "split_seed": int(render["split_seed"]),
        "counts": {split: len(values) for split, values in split_rows.items()},
        "scenes": {split: len({row["scene_id"] for row in values}) for split, values in split_rows.items()},
        "fixture_sources": fixture_source_counts(rows),
        "camera_position": {
            "axes": coordinates.CAMERA_AXES,
            "depth": "signed_distance_along_camera_forward_axis",
            "rows_with_image_plane_pixels": sum(1 for row in rows if has_image_plane_pixels(row)),
        },
        "intrinsic_passes": intrinsic_passes_payload(rows, render),
        "manifest_sha256": {
            split: sha256_file(output_root / "manifests" / f"{split}.jsonl") for split in split_rows
        },
    }
    (output_root / "dataset_release.json").write_text(
        json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, ensure_ascii=False), flush=True)


def with_camera_coordinates(row: dict) -> dict:
    """Attach OpenCV camera coordinates and image-plane data to one manifest row."""
    try:
        converted, _, _ = coordinates.convert_row(row, require_intrinsics=False)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot derive camera coordinates for {row.get('id')!r}: {error}") from error
    return converted


def has_image_plane_pixels(row: dict) -> bool:
    for field in ("point_lights", "in_scene_lights"):
        for light in row.get(field) or []:
            plane = light.get("image_plane") if isinstance(light, dict) else None
            if isinstance(plane, dict) and "u" in plane:
                return True
    return False


def fixture_source_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"scene_native": 0, "procedural_fallback": 0}
    for row in rows:
        for fixture in row.get("in_scene_lights", []):
            source = fixture.get("fixture_source")
            if source in counts:
                counts[source] += 1
    return counts


def intrinsic_passes_payload(rows: list[dict], render: dict) -> dict:
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
            "min_base_color_linked_ratio": float(render_passes.get("min_base_color_linked_ratio", 0.9)),
            "max_magenta_ratio": float(render_passes.get("max_magenta_ratio", 1e-4)),
            "min_textured_pixel_ratio": 0.02,
        },
    }


def load_yaml(path: str) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


if __name__ == "__main__":
    main()
