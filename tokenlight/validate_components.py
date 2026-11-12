"""Validate scene-only TokenLight components and manifests.

Checks that each rendered sample is complete and self-consistent:

* every declared component file exists and is non-empty (ambient, dark,
  diffuse, point lights, in-scene fixture, optional passes);
* the camera block carries intrinsics and the canonical frame matches it
  (via :mod:`datacompose.coordinates`, the single source of truth);
* light positions convert to camera space without disagreement;
* a scene never appears in more than one split;
* optional intrinsic-pass quality thresholds are met.

    python tokenlight/validate_components.py --config stage2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Run as a plain script (`python tokenlight/validate_components.py --config x.yaml`)
# from any working directory and from a renamed project folder.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
from _launcher import ensure_package  # noqa: E402

ensure_package()

from datacompose import coordinates, intrinsic  # noqa: E402


FIXTURE_SOURCES = {"scene_native", "procedural_fallback"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scene-only TokenLight components.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    from tqdm.auto import tqdm

    config = load_yaml(parse_args().config)
    root = Path(config["paths"]["dataset_root"]).expanduser()
    data_config = config.get("data", {})
    passes_options = data_config.get("passes") or {}
    quality = {
        "min_base_color_linked_ratio": float(
            passes_options.get("min_base_color_linked_ratio", data_config.get("min_base_color_linked_ratio", 0.9))
        ),
        "max_magenta_ratio": float(
            passes_options.get("max_magenta_ratio", data_config.get("max_magenta_ratio", 1e-4))
        ),
        "require_textures": bool(
            passes_options.get("require_textures", data_config.get("require_textures", False))
        ),
    }

    manifest_paths = sorted((root / "manifests").glob("*.jsonl"))
    if not manifest_paths:
        raise FileNotFoundError(f"no manifests under {root / 'manifests'}")

    errors: list[dict[str, Any]] = []
    scene_split: dict[str, str] = {}
    counts = {
        "scenes": 0,
        "rows": 0,
        "point_lights": 0,
        "diffuse": 0,
        "in_scene_lights": 0,
        "scene_native_fixtures": 0,
        "procedural_fallback_fixtures": 0,
        "pass_sets": 0,
        "components": 0,
    }

    for manifest in manifest_paths:
        split = manifest.stem
        for row in load_jsonl(manifest):
            counts["rows"] += 1
            label = str(row.get("id") or "<missing id>")
            scene_id = str(row.get("scene_id") or "")
            if not scene_id:
                errors.append({"id": label, "error": "missing_scene_id"})
                continue
            previous = scene_split.setdefault(scene_id, split)
            if previous != split:
                errors.append(
                    {
                        "id": label,
                        "scene_id": scene_id,
                        "error": "scene_split_leakage",
                        "splits": [previous, split],
                    }
                )
            counts["scenes"] = len(scene_split)
            try:
                counts["components"] += validate_row(row, root, quality, counts)
            except ValueError as error:
                errors.append({"id": label, "scene_id": scene_id, "error": str(error)})

    report = {
        "contract": "scene_only_v1",
        "dataset_root": str(root),
        "manifests": [path.name for path in manifest_paths],
        "counts": counts,
        "errors": errors,
        "error_count": len(errors),
    }
    (root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False))
    if errors:
        for row in errors[:20]:
            print(f"  ERROR {row}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} validation error(s); see {root / 'validation_report.json'}")
    print(f"OK: {counts['rows']} samples across {counts['scenes']} scenes")


def validate_row(row: dict[str, Any], root: Path, quality: dict[str, Any], counts: dict[str, Any]) -> int:
    """Validate one metadata row; returns the number of component files checked."""
    camera = row.get("camera")
    canonical = row.get("canonical")
    if not isinstance(camera, dict) or not isinstance(canonical, dict):
        raise ValueError("missing camera/canonical block")
    if canonical.get("position_axes") != coordinates.CANONICAL_AXES:
        raise ValueError(f"unsupported canonical.position_axes: {canonical.get('position_axes')!r}")
    if not row.get("lineage"):
        raise ValueError("missing lineage block")
    if coordinates.Intrinsics.from_camera(camera) is None:
        raise ValueError("camera block has no projection parameters")

    checked = 0
    checked += _require_file(root, row.get("ambient"), "ambient")
    checked += _require_file(root, row.get("dark"), "dark")

    points = row.get("point_lights") or []
    if not points:
        raise ValueError("sample has no point_lights")
    for index, light in enumerate(points):
        _require_vector(light.get("position"), 3, f"point_lights[{index}].position")
        checked += _require_file(root, light.get("path"), f"point_lights[{index}]")
    counts["point_lights"] += len(points)

    diffuse = row.get("diffuse") or []
    if not diffuse:
        raise ValueError("sample has no diffuse components")
    for index, component in enumerate(diffuse):
        checked += _require_file(root, component.get("path"), f"diffuse[{index}]")
    counts["diffuse"] += len(diffuse)

    fixtures = row.get("in_scene_lights") or []
    if not fixtures:
        raise ValueError("sample has no in_scene_lights entry")
    for index, fixture in enumerate(fixtures):
        source = fixture.get("fixture_source")
        if source not in FIXTURE_SOURCES:
            raise ValueError(f"in_scene_lights[{index}].fixture_source is invalid: {source!r}")
        counts[f"{source}_fixtures"] = counts.get(f"{source}_fixtures", 0) + 1
        counts["in_scene_lights"] += 1
        checked += _require_file(root, fixture.get("path"), f"in_scene_lights[{index}]")
        checked += _require_file(root, fixture.get("mask"), f"in_scene_lights[{index}].mask")

    # Coordinates: canonical.origin must equal camera.target and every light must
    # convert consistently (convert_row raises on disagreement).
    coordinates.convert_row(row, require_intrinsics=False)

    passes = row.get("passes")
    if isinstance(passes, dict):
        counts["pass_sets"] += 1
        checked += _require_file(root, passes.get("albedo"), "passes.albedo")
        checked += _require_file(root, passes.get("normal_camera"), "passes.normal_camera")
        checked += _require_file(root, passes.get("valid_mask"), "passes.valid_mask")
        _validate_pass_quality(passes.get("quality") or {}, quality)
    return checked


def _validate_pass_quality(reported: dict[str, Any], quality: dict[str, Any]) -> None:
    linked = reported.get("base_color_linked_ratio")
    if linked is not None and float(linked) < quality["min_base_color_linked_ratio"]:
        raise ValueError(
            f"passes quality: base_color_linked_ratio {float(linked):.3f} < "
            f"{quality['min_base_color_linked_ratio']}"
        )
    magenta = reported.get("magenta_ratio")
    if magenta is not None and float(magenta) > quality["max_magenta_ratio"]:
        raise ValueError(
            f"passes quality: magenta_ratio {float(magenta):.3g} > {quality['max_magenta_ratio']}"
        )
    if quality["require_textures"]:
        textured = reported.get("textured_pixel_ratio")
        if textured is not None and float(textured) <= 0.0:
            raise ValueError("passes quality: require_textures is set but no texture was sampled")


def validate_passes(
    root: Path,
    scene: dict[str, Any],
    ambient: np.ndarray,
    counts: dict[str, Any],
    *,
    require_textures: bool = False,
) -> None:
    """Validate one sample's intrinsic pass set (albedo / normal / irradiance).

    ``ambient`` is the ambient radiance image; the derived irradiance must be
    non-negative and the normal pass must be unit length wherever it is valid.
    Problems are raised as ``ValueError``; cosmetic issues (a magenta albedo when
    textures are not required) only append to ``counts["pass_warnings"]``.
    """
    passes = scene.get("passes")
    if not isinstance(passes, dict):
        raise ValueError("passes block must be a mapping")
    for key in ("albedo", "normal_camera", "valid_mask"):
        if not passes.get(key):
            raise ValueError(f"passes 缺少字段 {key}")

    albedo = _load_array(root, passes["albedo"], "passes.albedo")
    normal = _load_array(root, passes["normal_camera"], "passes.normal_camera")
    valid = _load_array(root, passes["valid_mask"], "passes.valid_mask")
    if albedo.shape != normal.shape:
        raise ValueError(
            f"passes shape mismatch: albedo {albedo.shape} vs normal {normal.shape}"
        )
    if valid.shape != albedo.shape[:2]:
        raise ValueError(f"passes shape mismatch: valid_mask {valid.shape} vs {albedo.shape[:2]}")

    mask = valid.astype(bool)
    if mask.any():
        lengths = np.linalg.norm(normal[mask], axis=-1)
        if not np.all(np.abs(lengths - 1.0) <= 1e-3):
            raise ValueError("passes.normal_camera is not unit length on valid pixels")

    irradiance_key = None
    irradiance_block = passes.get("irradiance")
    if isinstance(irradiance_block, dict) and irradiance_block.get("ambient"):
        irradiance_key = "ambient"
    if irradiance_key is not None:
        derived = _load_array(root, irradiance_block[irradiance_key], "passes.irradiance.ambient")
        if derived.shape != albedo.shape:
            raise ValueError(
                f"passes shape mismatch: irradiance {derived.shape} vs albedo {albedo.shape}"
            )
        if float(derived.min()) < -1e-6:
            raise ValueError("passes.irradiance contains negative radiance")
    elif ambient is None:
        raise ValueError("passes.irradiance missing and no ambient radiance supplied")

    quality = passes.get("quality") or {}
    counts.setdefault("pass_warnings", [])
    # Blender renders missing textures as magenta; measure it from the pixels
    # rather than trusting the renderer's own summary.
    magenta = _magenta_ratio(albedo)
    if magenta > 0.0:
        if require_textures:
            raise ValueError(
                f"passes.albedo is {magenta:.3g} magenta while require_textures is set"
            )
        counts["pass_warnings"].append(
            f"magenta albedo ratio {magenta:.3g} (textures not required)"
        )
    reported = quality.get("base_color_linked_ratio")
    if reported is not None and float(reported) < 0.0:
        raise ValueError("passes.quality.base_color_linked_ratio must be non-negative")
    counts["pass_sets"] = counts.get("pass_sets", 0) + 1


def _magenta_ratio(albedo: np.ndarray) -> float:
    """Fraction of pixels that are the magenta "missing texture" placeholder."""
    if albedo.ndim != 3 or albedo.shape[-1] < 3 or albedo.size == 0:
        return 0.0
    red = albedo[..., 0]
    green = albedo[..., 1]
    blue = albedo[..., 2]
    mask = (red > 0.99) & (green < 0.01) & (blue > 0.99)
    return float(mask.sum()) / float(albedo.shape[0] * albedo.shape[1])


def _load_array(root: Path, relative: Any, label: str) -> np.ndarray:
    """Load a pass array from ``.npy`` or (when OpenCV is available) ``.exr``."""
    path = root / str(relative)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import cv2  # noqa: PLC0415 - optional dependency, only needed for EXR/PNG passes

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{label} could not be decoded: {path}")
    if image.ndim == 2:
        return image.astype(np.float32)
    return image[..., ::-1].astype(np.float32)  # BGR -> RGB
    if not relative:
        raise ValueError(f"{label} has no path")
    path = root / str(relative)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {path}")
    return 1


def _require_file(root: Path, relative: Any, label: str) -> int:
    if not relative:
        raise ValueError(f"{label} has no path")
    path = root / str(relative)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {path}")
    return 1


def _require_vector(value: Any, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a list of {length} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain numbers")
    if not all(np.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must be finite")


def load_yaml(path: str) -> dict:
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


if __name__ == "__main__":
    main()
