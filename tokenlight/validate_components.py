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
from datacompose.tokenlight import validator_checks  # noqa: E402


FIXTURE_SOURCES = {"scene_native", "procedural_fallback"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate scene-only TokenLight components.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pixel-checks",
        action="store_true",
        help=(
            "also decode albedo/normal/irradiance pixels and verify the "
            "albedo*irradiance identity (requires opencv-python)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = Path(config["paths"]["dataset_root"]).expanduser()
    data_config = config.get("data", {})
    # ``data.passes`` is authoritative, but older dataset configs put the three
    # quality switches directly under ``data``; keep accepting both spellings.
    passes_options = dict(data_config.get("passes") or {})
    for key in ("max_magenta_ratio", "min_base_color_linked_ratio", "require_textures"):
        if key not in passes_options and key in data_config:
            passes_options[key] = data_config[key]
    options = validator_checks.normalized_passes_options(passes_options)
    quality = {
        "min_base_color_linked_ratio": float(options["min_base_color_linked_ratio"]),
        "max_magenta_ratio": options["max_magenta_ratio"],
        "require_textures": bool(options["require_textures"]),
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
        "pass_pixel_sets": 0,
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
                counts["components"] += validate_row(
                    row, root, quality, counts, pixel_checks=args.pixel_checks
                )
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


def validate_row(
    row: dict[str, Any],
    root: Path,
    quality: dict[str, Any],
    counts: dict[str, Any],
    *,
    pixel_checks: bool = False,
) -> int:
    """Validate one metadata row; returns the number of component files checked."""
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("missing scene_id")
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
        enabled = passes.get("enabled")
        if not isinstance(enabled, dict):
            # Samples rendered before the ``enabled`` block existed always had
            # every pass, so keep the strict legacy requirements for them.
            enabled = {"albedo": True, "normal": True, "irradiance": True}
        audit_relative = passes.get("material_audit")
        checked += _require_file(root, audit_relative, "passes.material_audit")
        try:
            audit_report = json.loads(
                (root / str(audit_relative)).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"passes.material_audit cannot be decoded: {error}") from error
        audit_problems = validator_checks.validate_material_audit(audit_report)
        if audit_problems:
            raise ValueError("; ".join(audit_problems))
        use_albedo = bool(enabled.get("albedo", True))
        use_normal = bool(enabled.get("normal", True))
        if use_albedo:
            checked += _require_file(root, passes.get("albedo"), "passes.albedo")
        if use_normal:
            checked += _require_file(root, passes.get("normal_camera"), "passes.normal_camera")
        if use_albedo and use_normal:
            checked += _require_file(root, passes.get("valid_mask"), "passes.valid_mask")
        if passes.get("normal_world"):
            checked += _require_file(root, passes["normal_world"], "passes.normal_world")
        if bool(enabled.get("irradiance", True)):
            irradiance = passes.get("irradiance")
            if not isinstance(irradiance, dict) or not irradiance.get("ambient"):
                raise ValueError("passes.irradiance.ambient is missing")
            expected = validator_checks.expected_irradiance_keys(len(points), len(fixtures))
            missing = sorted(expected - set(irradiance))
            if missing:
                raise ValueError(
                    f"passes.irradiance is missing component keys: {missing}"
                )
            for label, relative in irradiance.items():
                checked += _require_file(root, relative, f"passes.irradiance.{label}")
        _validate_pass_quality(passes.get("quality") or {}, quality)
        if pixel_checks:
            _validate_pixel_passes(root, row, passes, counts, quality)
    return checked


def _validate_pixel_passes(
    root: Path,
    row: dict[str, Any],
    passes: dict[str, Any],
    counts: dict[str, Any],
    quality: dict[str, Any],
) -> None:
    """Decode the pass products and verify normals, radiance and the identity."""
    components: dict[str, np.ndarray] = {}
    ambient = None
    if row.get("ambient"):
        ambient = _load_array(root, row["ambient"], "ambient")
        components["ambient"] = ambient
    dark = _load_array(root, row["dark"], "dark") if row.get("dark") else None
    for index, light in enumerate(row.get("point_lights") or []):
        if isinstance(light, dict) and light.get("path"):
            components[f"point_{index:03d}"] = _load_array(
                root, light["path"], f"point_lights[{index}]"
            )
    for index, fixture in enumerate(row.get("in_scene_lights") or []):
        if isinstance(fixture, dict) and fixture.get("path"):
            components[f"fixture_{index:03d}"] = _load_array(
                root, fixture["path"], f"in_scene_lights[{index}]"
            )
    pixel_counts: dict[str, Any] = {}
    validate_passes(
        root,
        row,
        ambient,
        pixel_counts,
        require_textures=bool(quality.get("require_textures", False)),
        dark=dark,
        components=components,
    )
    counts["pass_pixel_sets"] = counts.get("pass_pixel_sets", 0) + 1
    if pixel_counts.get("pass_warnings"):
        counts.setdefault("pass_warnings", []).extend(pixel_counts["pass_warnings"])


def _validate_pass_quality(reported: dict[str, Any], quality: dict[str, Any]) -> None:
    linked = reported.get("base_color_linked_ratio")
    if linked is not None and float(linked) < quality["min_base_color_linked_ratio"]:
        raise ValueError(
            f"passes quality: base_color_linked_ratio {float(linked):.3f} < "
            f"{quality['min_base_color_linked_ratio']}"
        )
    problems = validator_checks.pass_quality_problems(
        reported,
        max_magenta_ratio=quality["max_magenta_ratio"],
        require_textures=bool(quality["require_textures"]),
    )
    if problems:
        raise ValueError("passes quality: " + "; ".join(problems))


def validate_passes(
    root: Path,
    scene: dict[str, Any],
    ambient: np.ndarray | None,
    counts: dict[str, Any],
    *,
    require_textures: bool = False,
    dark: np.ndarray | None = None,
    components: dict[str, np.ndarray] | None = None,
    max_median_relative_error: float = 0.05,
    min_identity_albedo: float = 0.1,
) -> None:
    """Validate one sample's intrinsic pass set (albedo / normal / irradiance).

    ``ambient`` is the ambient radiance image; the derived irradiance must be
    non-negative and the normal pass must be unit length wherever it is valid.
    When ``dark`` and ``components`` are supplied, the documented identity
    ``albedo * irradiance_L ~= component_L - dark`` is also checked per lighting
    component.  Problems are raised as ``ValueError``; cosmetic issues (a magenta
    albedo when textures are not required) only append to
    ``counts["pass_warnings"]``.
    """
    passes = scene.get("passes")
    if not isinstance(passes, dict):
        raise ValueError("passes block must be a mapping")
    enabled = passes.get("enabled")
    if not isinstance(enabled, dict):
        enabled = {"albedo": True, "normal": True, "irradiance": True}
    use_albedo = bool(enabled.get("albedo", True))
    use_normal = bool(enabled.get("normal", True))
    use_irradiance = bool(enabled.get("irradiance", True))
    if use_irradiance and not use_albedo:
        raise ValueError("passes.irradiance=true requires passes.albedo=true")

    albedo = normal = valid = None
    if use_albedo:
        if not passes.get("albedo"):
            raise ValueError("passes 缺少字段 albedo")
        albedo = _load_array(root, passes["albedo"], "passes.albedo")
    if use_normal:
        if not passes.get("normal_camera"):
            raise ValueError("passes 缺少字段 normal_camera")
        normal = _load_array(root, passes["normal_camera"], "passes.normal_camera")
    if use_albedo and use_normal:
        if not passes.get("valid_mask"):
            raise ValueError("passes 缺少字段 valid_mask")
        valid = _load_array(root, passes["valid_mask"], "passes.valid_mask")
    if albedo is not None and normal is not None and albedo.shape != normal.shape:
        raise ValueError(
            f"passes shape mismatch: albedo {albedo.shape} vs normal {normal.shape}"
        )
    if valid is not None and albedo is not None:
        if valid.shape != albedo.shape[:2]:
            raise ValueError(
                f"passes shape mismatch: valid_mask {valid.shape} vs {albedo.shape[:2]}"
            )

    if normal is not None:
        mask = (
            valid.astype(bool)
            if valid is not None
            else np.linalg.norm(normal, axis=-1) > 0.0
        )
        if mask.any():
            lengths = np.linalg.norm(normal[mask], axis=-1)
            if not np.all(np.abs(lengths - 1.0) <= 1e-3):
                raise ValueError("passes.normal_camera is not unit length on valid pixels")

    irradiance_key = None
    irradiance_block = passes.get("irradiance")
    if use_irradiance and isinstance(irradiance_block, dict) and irradiance_block.get("ambient"):
        irradiance_key = "ambient"
    if irradiance_key is not None:
        derived = _load_array(root, irradiance_block[irradiance_key], "passes.irradiance.ambient")
        if albedo is not None and derived.shape != albedo.shape:
            raise ValueError(
                f"passes shape mismatch: irradiance {derived.shape} vs albedo {albedo.shape}"
            )
        if float(derived.min()) < -1e-6:
            raise ValueError("passes.irradiance contains negative radiance")
    elif use_irradiance and ambient is None:
        raise ValueError("passes.irradiance missing and no ambient radiance supplied")

    if use_irradiance and albedo is not None and dark is not None:
        _check_identity(
            root,
            albedo,
            irradiance_block,
            dark,
            components or {},
            valid,
            counts,
            max_median_relative_error=max_median_relative_error,
            min_albedo=min_identity_albedo,
        )

    quality = passes.get("quality") or {}
    counts.setdefault("pass_warnings", [])
    # Blender renders missing textures as magenta; measure it from the pixels
    # rather than trusting the renderer's own summary.
    if albedo is not None:
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


def _check_identity(
    root: Path,
    albedo: np.ndarray,
    irradiance_block: dict[str, Any],
    dark: np.ndarray,
    components: dict[str, np.ndarray],
    valid: np.ndarray | None,
    counts: dict[str, Any],
    *,
    max_median_relative_error: float,
    min_albedo: float,
) -> None:
    """Verify ``albedo * irradiance_L ~= component_L - dark`` per component."""
    reports: dict[str, Any] = {}
    failures: list[str] = []
    for label, relative in irradiance_block.items():
        radiance = components.get(label)
        if radiance is None:
            continue
        derived = _load_array(root, relative, f"passes.irradiance.{label}")
        report = intrinsic.identity_error(
            albedo,
            derived,
            radiance,
            dark,
            valid=valid,
            min_albedo=min_albedo,
        )
        if report is None:
            continue
        reports[label] = report
        if report["median_relative_error"] > float(max_median_relative_error):
            failures.append(
                f"{label}: median relative error "
                f"{report['median_relative_error']:.3g} > {float(max_median_relative_error):.3g}"
            )
    if reports:
        counts["pass_identity"] = reports
    if failures:
        raise ValueError("albedo*irradiance identity failed: " + "; ".join(failures))


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
    try:
        import cv2  # noqa: PLC0415 - optional dependency, only needed for EXR/PNG passes
    except ImportError as error:
        raise ValueError(
            f"{label} needs opencv-python to decode {path.suffix} files: {error}"
        ) from error

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{label} could not be decoded: {path}")
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 4:
        image = image[..., :3]
    return image[..., ::-1].astype(np.float32)  # BGR -> RGB


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
