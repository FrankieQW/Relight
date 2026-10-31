from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import ProjectConfig
from .io_utils import dump_json_atomic, dump_jsonl_atomic, load_json, sha256_file, utc_now
from .schemas import ObjectDocument, PreparedGeometryDocument


_AXES = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


def prepare_geometry(config: ProjectConfig) -> dict[str, Any]:
    object_path = config.path("object_output")
    objects = ObjectDocument.model_validate(load_json(object_path)).model_dump(mode="json")
    m4 = config.section("m4")
    category_path = _project_path(config, str(m4["category_dimensions_file"]))
    override_path = _project_path(config, str(m4["object_overrides_file"]))
    category_document = _load_yaml_mapping(category_path)
    override_document = _load_yaml_mapping(override_path)
    category_rules = category_document.get("categories", {})
    object_overrides = override_document.get("objects", {})
    if not isinstance(category_rules, dict) or not isinstance(object_overrides, dict):
        raise TypeError("geometry category and object override roots must be mappings")

    root_env = str(m4.get("object_root_env") or objects["object_root_env"])
    root_value = os.environ.get(root_env)
    if not root_value:
        raise EnvironmentError(f"required object root environment variable is not set: {root_env}")
    object_root = Path(root_value).expanduser().resolve()
    if not object_root.is_dir():
        raise NotADirectoryError(f"object root does not exist: {object_root}")

    allowlist = {
        str(value).strip().lower()
        for value in config.section("object").get("license_allowlist", [])
        if str(value).strip()
    }
    enforce_allowlist = bool(m4.get("enforce_license_allowlist", True))
    if enforce_allowlist and not allowlist:
        raise ValueError(
            "object.license_allowlist is empty while m4.enforce_license_allowlist=true"
        )

    geometries: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in objects["objects"]:
        uid = row["uid"]
        category = row["primary_category_normalized"]
        override = object_overrides.get(uid, {})
        category_rule = category_rules.get(category)
        try:
            if override is not None and not isinstance(override, dict):
                raise ValueError("object override must be a mapping")
            override = override or {}
            if bool(override.get("disabled", False)):
                raise ValueError("disabled_by_object_override")
            if category_rule is not None and not isinstance(category_rule, dict):
                raise ValueError("category geometry rule must be a mapping")
            rule = {**(category_rule or {}), **override}
            if not category_rule and not override:
                raise ValueError("missing_category_dimensions")
            target_dimensions = _dimensions(rule.get("target_dimensions"))
            up_axis = _axis(rule.get("up_axis"), "up_axis")
            front_axis = _axis(rule.get("front_axis"), "front_axis")
            contact_axis = _axis(rule.get("contact_axis", "-Z"), "contact_axis")
            if up_axis.lstrip("+-") == front_axis.lstrip("+-"):
                raise ValueError("up_axis and front_axis must use different dimensions")
            expected_contact = ("-" if up_axis.startswith("+") else "+") + up_axis[-1]
            if contact_axis != expected_contact:
                raise ValueError(
                    f"contact_axis must oppose up_axis ({up_axis}); expected {expected_contact}"
                )
            fit_mode = str(rule.get("fit_mode", "uniform_fit"))
            if fit_mode not in {"uniform_fit", "exact_dimensions"}:
                raise ValueError(f"unsupported fit_mode: {fit_mode}")

            asset_path, asset_relative = _resolve_asset_path(row, object_root)

            license_name = str(row.get("license") or "").strip()
            allowed = not enforce_allowlist or license_name.lower() in allowlist
            if not allowed:
                raise ValueError(f"license_not_allowlisted:{license_name or 'missing'}")
            geometries.append(
                {
                    "object_uid": uid,
                    "object_category": category,
                    "asset_path": asset_relative,
                    "asset_digest": sha256_file(asset_path),
                    "target_dimensions": target_dimensions,
                    "up_axis": up_axis,
                    "front_axis": front_axis,
                    "contact_axis": contact_axis,
                    "fit_mode": fit_mode,
                    "config_source": "object_override" if override else "category_default",
                    "license": {
                        "name": row.get("license"),
                        "raw": row.get("license_raw"),
                        "author": row.get("author", {}),
                        "source_uri": row.get("source_uri"),
                        "decision": "allowed",
                    },
                }
            )
        except Exception as error:  # noqa: BLE001 - quarantine preserves every asset failure.
            quarantine.append(
                {
                    "object_uid": uid,
                    "object_category": category,
                    "asset_path": row.get("canonical_path"),
                    "reason": str(error),
                }
            )

    document = PreparedGeometryDocument(
        schema_version=str(config.section("project").get("schema_version", "1.0")),
        generated_at=utc_now(),
        generator_version=__version__,
        config_digest=config.digest,
        object_digest=sha256_file(object_path),
        geometry_config_digests={
            "category_dimensions": sha256_file(category_path),
            "object_overrides": sha256_file(override_path),
        },
        object_root_env=root_env,
        geometries=sorted(geometries, key=lambda item: item["object_uid"]),
        quarantine=sorted(quarantine, key=lambda item: item["object_uid"]),
        stats={
            "objects": len(objects["objects"]),
            "prepared": len(geometries),
            "quarantined": len(quarantine),
        },
    ).model_dump(mode="json")
    dump_json_atomic(config.path("prepared_geometry_output"), document)
    dump_jsonl_atomic(config.path("geometry_quarantine"), document["quarantine"])
    if not document["geometries"]:
        raise RuntimeError("geometry preparation produced zero usable objects")
    return document


def _dimensions(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("target_dimensions must contain exactly three meter values")
    dimensions = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in dimensions):
        raise ValueError("target_dimensions must contain finite positive values")
    return dimensions


def _resolve_asset_path(row: dict[str, Any], object_root: Path) -> tuple[Path, str]:
    """Resolve an asset using inventory_path as a bounded legacy fallback."""
    candidates: list[str] = []
    for key in ("canonical_path", "inventory_path"):
        value = str(row.get(key) or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    if not candidates:
        raise ValueError("object row has neither canonical_path nor inventory_path")

    root = object_root.resolve()
    attempted: list[str] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        resolved = (path if path.is_absolute() else root / path).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            attempted.append(f"{candidate} (outside OBJECT_ROOT)")
            continue
        attempted.append(str(resolved))
        if resolved.is_file():
            return resolved, relative
    raise FileNotFoundError("object asset does not exist; attempted: " + ", ".join(attempted))


def _axis(value: Any, field: str) -> str:
    axis = str(value or "").upper()
    if axis not in _AXES:
        raise ValueError(f"{field} must be one of {sorted(_AXES)}")
    return axis


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _project_path(config: ProjectConfig, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config.root / path).resolve()
