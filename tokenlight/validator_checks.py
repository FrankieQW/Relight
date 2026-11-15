"""Framework-independent checks shared by render-time tests and validators."""
from __future__ import annotations

from typing import Any, Mapping


CONFIG_DEFAULTS = {
    "enabled": False,
    "albedo": True,
    "normal": True,
    "irradiance": True,
    "irradiance_mode": "derived",
    "normal_space": "camera",
    "save_normal_world": False,
    "pass_samples": 1,
    "eps": 0.001,
    "denoise_passes": False,
    "require_textures": False,
    "min_base_color_linked_ratio": 0.9,
    "max_magenta_ratio": None,
}


def normalized_passes_options(value: Any) -> dict[str, Any]:
    """Merge pass settings with defaults without mutating the input mapping."""
    if value is None:
        source = {}
    elif isinstance(value, Mapping):
        source = value
    else:
        raise TypeError("render.passes must be a mapping")
    options = {**CONFIG_DEFAULTS, **source}
    if options["irradiance_mode"] not in {"derived", "diffuse_radiance_over_albedo"}:
        raise ValueError("render.passes.irradiance_mode must be 'derived'")
    if options["normal_space"] != "camera":
        raise ValueError("render.passes.normal_space must be 'camera'")
    if float(options["eps"]) <= 0.0:
        raise ValueError("render.passes.eps must be positive")
    if int(options["pass_samples"]) < 1:
        raise ValueError("render.passes.pass_samples must be at least 1")
    ratio = float(options["min_base_color_linked_ratio"])
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("render.passes.min_base_color_linked_ratio must be in [0, 1]")
    return options


def expected_irradiance_keys(point_count: int, fixture_count: int) -> set[str]:
    keys = {"ambient"}
    keys.update(f"point_{index:03d}" for index in range(point_count))
    keys.update(f"fixture_{index:03d}" for index in range(fixture_count))
    return keys


def pass_quality_problems(
    quality: Mapping[str, Any],
    *,
    max_magenta_ratio: float | None,
    require_textures: bool,
) -> list[str]:
    problems: list[str] = []
    magenta = float(quality.get("magenta_ratio") or 0.0)
    if max_magenta_ratio is not None and magenta > float(max_magenta_ratio):
        problems.append(f"albedo magenta ratio {magenta:.3g} > {float(max_magenta_ratio):.3g}")
    if require_textures and float(quality.get("textured_pixel_ratio") or 0.0) <= 0.0:
        problems.append("require_textures=true but albedo has no textured pixels")
    return problems


def validate_material_audit(report: Any, *, textured_materials: int | None = None) -> list[str]:
    """Check the fixed material-audit schema and ratio denominator."""
    if not isinstance(report, Mapping):
        return ["material audit is not a mapping"]
    required = {
        "materials",
        "textured",
        "textured_materials",
        "base_color_linked",
        "base_color_linked_ratio",
        "unknown_materials",
        "linked_without_image",
        "missing_image_files",
        "missing_examples",
        "rows",
    }
    missing = sorted(required - set(report))
    if missing:
        return [f"material audit missing fields: {missing}"]
    rows = report.get("rows")
    if not isinstance(rows, list):
        return ["material audit rows must be a list"]
    expected = int(report["textured_materials"])
    if textured_materials is not None and expected != int(textured_materials):
        return ["material audit textured_materials disagrees with metadata"]
    if expected and float(report["base_color_linked_ratio"] or 0.0) > 1.0:
        return ["material audit base_color_linked_ratio exceeds 1"]
    return []


__all__ = [
    "CONFIG_DEFAULTS",
    "expected_irradiance_keys",
    "normalized_passes_options",
    "pass_quality_problems",
    "validate_material_audit",
]