from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datacompose import coordinates, intrinsic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TokenLight linear components and UID-safe splits.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    from tqdm.auto import tqdm

    config = load_yaml(parse_args().config)
    root = Path(config["paths"]["dataset_root"]).expanduser()
    data_config = config["data"]
    require_composition = bool(data_config.get("require_composition_contract", False))
    split_profile = str(data_config.get("split_profile", "object-held-out"))
    if split_profile not in {"object-held-out", "scene-held-out"}:
        raise ValueError("data.split_profile must be object-held-out or scene-held-out")
    manifests = {
        split: config["paths"].get(f"{split}_manifest")
        for split in ("train", "validation", "test")
        if config["paths"].get(f"{split}_manifest")
    }
    errors: list[dict[str, Any]] = []
    seen_uids: dict[str, str] = {}
    seen_base_scenes: dict[str, str] = {}
    counts = {
        "scenes": 0,
        "point_lights": 0,
        "diffuse": 0,
        "in_scene_lights": 0,
        "scene_native_fixtures": 0,
        "procedural_fallback_fixtures": 0,
        "pass_sets": 0,
        "pass_warnings": [],
        "task_eligible_scenes": {task: 0 for task in data_config["tasks"]},
    }
    passes_options = data_config.get("passes") or {}
    pass_options = {
        "require_textures": bool(
            passes_options.get("require_textures", data_config.get("require_textures", False))
        ),
        "min_base_color_linked_ratio": float(
            passes_options.get("min_base_color_linked_ratio", 0.9)
        ),
        "max_magenta_ratio": float(passes_options.get("max_magenta_ratio", 1e-4)),
        "min_textured_pixel_ratio": float(
            passes_options.get("min_textured_pixel_ratio", 0.02)
        ),
    }
    manifest_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for split, manifest_value in manifests.items():
        manifest = Path(manifest_value).expanduser()
        if not manifest.is_file():
            errors.append({"split": split, "error": "missing_manifest", "path": str(manifest)})
            continue
        manifest_rows.append((split, read_jsonl(manifest)))
    total_scenes = sum(len(rows) for _, rows in manifest_rows)
    with tqdm(total=total_scenes, desc="validate components", unit="scene", dynamic_ncols=True) as progress:
        for split, rows in manifest_rows:
            for row in rows:
                counts["scenes"] += 1
                uid = str(row.get("asset_uid") or Path(row.get("asset", row["id"])).stem)
                previous = seen_uids.setdefault(uid, split)
                if previous != split and split_profile == "object-held-out":
                    errors.append({"scene_id": row["id"], "asset_uid": uid, "error": "asset_uid_split_leakage", "splits": [previous, split]})
                base_scene_id = str(row.get("base_scene_id", ""))
                if base_scene_id:
                    previous_scene = seen_base_scenes.setdefault(base_scene_id, split)
                    if previous_scene != split and split_profile == "scene-held-out":
                        errors.append(
                            {
                                "scene_id": row["id"],
                                "base_scene_id": base_scene_id,
                                "error": "base_scene_split_leakage",
                                "splits": [previous_scene, split],
                            }
                        )
                try:
                    validate_scene(
                        root,
                        row,
                        counts,
                        require_composition=require_composition,
                        pass_options=pass_options,
                    )
                except Exception as error:
                    errors.append({"scene_id": row.get("id"), "split": split, "error": type(error).__name__, "detail": str(error)})
                progress.update(1)
                progress.set_postfix(split=split, errors=len(errors))
    for task in data_config["tasks"]:
        if counts["task_eligible_scenes"][task] == 0:
            errors.append({"error": "configured_task_has_no_eligible_scene", "task": task})
    report_path = root / "validation_errors.jsonl"
    write_jsonl(report_path, errors)
    summary = {
        **counts,
        "split_profile": split_profile,
        "require_composition_contract": require_composition,
        "errors": len(errors),
        "error_log": str(report_path),
    }
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError(f"数据验证发现 {len(errors)} 个错误，详见 {report_path}")


def validate_scene(
    root: Path,
    scene: dict[str, Any],
    counts: dict[str, Any],
    *,
    require_composition: bool,
    pass_options: dict[str, Any] | None = None,
) -> None:
    for key in ("id", "asset", "ambient", "dark", "point_lights", "diffuse", "camera", "canonical"):
        if key not in scene:
            raise ValueError(f"manifest 缺少字段 {key}")
    if require_composition:
        validate_composition_contract(scene)
    ambient = read_linear(resolve(root, scene["ambient"]))
    dark = read_linear(resolve(root, scene["dark"]))
    if ambient.shape != dark.shape:
        raise ValueError(f"ambient/dark shape 不一致: {ambient.shape} vs {dark.shape}")
    if ambient.shape[2] < 3:
        raise ValueError(f"图像不是 RGB: {ambient.shape}")
    point_energies = []
    for light in scene["point_lights"]:
        require_vector(light.get("position"), 3, "point_light.position")
        if "renderer_position" in light:
            require_vector(light.get("renderer_position"), 3, "point_light.renderer_position")
        if require_composition:
            require_camera_position(scene, light, "point_light")
        component = read_linear(resolve(root, light["path"]))
        require_shape(component, ambient, light["path"])
        point_energies.append(float(np.maximum(component - dark, 0).mean()))
        counts["point_lights"] += 1
    if point_energies and max(point_energies) <= 1e-8:
        raise ValueError("所有 point-light contribution 均为全黑")
    diffuse_means = []
    for item in scene["diffuse"]:
        component = read_linear(resolve(root, item["path"]))
        require_shape(component, ambient, item["path"])
        diffuse_means.append(float(component.mean()))
        counts["diffuse"] += 1
    if len(diffuse_means) >= 2 and max(diffuse_means) - min(diffuse_means) <= 1e-8:
        raise ValueError("diffuse levels 没有可测量差异")
    for fixture in scene.get("in_scene_lights", []):
        if not require_composition:
            validate_fixture_contract(fixture)
        if require_composition:
            require_camera_position(scene, fixture, "in_scene_light")
        component = read_linear(resolve(root, fixture["path"]))
        require_shape(component, ambient, fixture["path"])
        contribution = np.maximum(component - dark, 0)
        if float(contribution.mean()) <= 1e-8:
            raise ValueError(f"in-scene light contribution 全黑: {fixture['path']}")
        mask = read_mask(resolve(root, fixture["mask"]))
        if mask.shape != ambient.shape[:2]:
            raise ValueError(f"fixture mask shape 不一致: {mask.shape} vs {ambient.shape[:2]}")
        if float(mask.max()) <= 0:
            raise ValueError(f"fixture mask 为空: {fixture['mask']}")
        source = fixture["fixture_source"]
        counts[f"{source}_fixtures"] += 1
        counts["in_scene_lights"] += 1
    supported = {
        "ambient_scale": bool(scene.get("ambient")),
        "global_diffuse": len(scene.get("diffuse", [])) >= 2,
        "add_light": bool(scene.get("point_lights")),
        "in_scene_light": bool(scene.get("in_scene_lights")),
    }
    for task in counts["task_eligible_scenes"]:
        counts["task_eligible_scenes"][task] += int(supported.get(task, False))
    if scene.get("passes") is not None:
        validate_passes(root, scene, ambient, counts, **(pass_options or {}))


def validate_passes(
    root: Path,
    scene: dict[str, Any],
    ambient: np.ndarray,
    counts: dict[str, Any],
    *,
    require_textures: bool = False,
    min_base_color_linked_ratio: float = 0.9,
    max_magenta_ratio: float = 1e-4,
    min_textured_pixel_ratio: float = 0.02,
) -> None:
    """Validate the intrinsic pass set (albedo / normals / irradiance / mask).

    Structural problems always raise.  Quality problems raise only when
    ``require_textures`` is set; otherwise they are collected in
    ``counts["pass_warnings"]`` so a dataset can still be inspected.
    """
    passes = scene["passes"]
    if not isinstance(passes, dict):
        raise ValueError("passes 必须是 mapping")
    for key in ("albedo", "normal_camera", "valid_mask"):
        if not passes.get(key):
            raise ValueError(f"passes 缺少字段 {key}")
    definition = passes.get("definition") or {}
    eps = float(definition.get("eps", 1e-3))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("passes.definition.eps 非法")

    albedo = read_linear(resolve(root, passes["albedo"]))
    require_shape(albedo, ambient, passes["albedo"])
    if not np.isfinite(albedo).all():
        raise ValueError("albedo 含 NaN/Inf")
    if float(albedo.min()) < -1e-3 or float(albedo.max()) > 1.0 + 1e-3:
        raise ValueError(
            f"albedo 超出 [-1e-3, 1+1e-3]: [{float(albedo.min()):.4g}, {float(albedo.max()):.4g}]"
        )

    normal = read_linear(resolve(root, passes["normal_camera"]))
    require_shape(normal, ambient, passes["normal_camera"])
    valid = read_mask(resolve(root, passes["valid_mask"]))
    if valid.shape != ambient.shape[:2]:
        raise ValueError(f"valid_mask shape 不一致: {valid.shape} vs {ambient.shape[:2]}")
    valid_bool = valid.astype(bool)
    if not valid_bool.any():
        raise ValueError("valid_mask 全零")
    lengths = np.linalg.norm(normal[valid_bool], axis=1)
    if lengths.size and np.any(np.abs(lengths - 1.0) > 1e-3):
        raise ValueError("valid 像素上 normal 不是单位向量")

    irradiance_files = passes.get("irradiance") or {}
    if not isinstance(irradiance_files, dict) or not irradiance_files:
        raise ValueError("passes.irradiance 不能为空")
    for label, relative in irradiance_files.items():
        irradiance = read_linear(resolve(root, relative))
        require_shape(irradiance, ambient, relative)
        if not np.isfinite(irradiance).all():
            raise ValueError(f"irradiance {label} 含 NaN/Inf")
        if float(irradiance.min()) < -1e-6:
            raise ValueError(f"irradiance {label} 出现负值")

    counts.setdefault("pass_sets", 0)
    warnings = counts.setdefault("pass_warnings", [])
    scene_id = scene.get("id")
    measured = intrinsic.albedo_quality(albedo)
    if float(measured["magenta_ratio"]) > float(max_magenta_ratio):
        message = (
            f"{scene_id}: albedo magenta_ratio={measured['magenta_ratio']:.3g} "
            f"> {max_magenta_ratio:.3g}"
        )
        if require_textures:
            raise ValueError(message)
        warnings.append(message)

    stored = passes.get("quality") or {}
    textured_ratio = float(
        stored.get("textured_pixel_ratio", measured["textured_pixel_ratio"])
    )
    if textured_ratio < float(min_textured_pixel_ratio):
        message = (
            f"{scene_id}: textured_pixel_ratio={textured_ratio:.3g} "
            f"< {min_textured_pixel_ratio:.3g}"
        )
        if require_textures:
            raise ValueError(message)
        warnings.append(message)

    audit_relative = passes.get("material_audit")
    if audit_relative:
        audit = read_json(resolve(root, audit_relative))
        stored_ratio = stored.get("base_color_linked_ratio")
        audit_ratio = audit.get("base_color_linked_ratio")
        if (
            stored_ratio is not None
            and audit_ratio is not None
            and abs(float(stored_ratio) - float(audit_ratio)) > 1e-6
        ):
            raise ValueError(
                "passes.quality.base_color_linked_ratio 与 material_audit.json 不一致"
            )
        for problem in intrinsic.evaluate_material_audit(
            audit,
            min_base_color_linked_ratio=min_base_color_linked_ratio,
            require_textures=True,
        ):
            message = f"{scene_id}: {problem}"
            if require_textures:
                raise ValueError(message)
            warnings.append(message)
    counts["pass_sets"] += 1


def validate_composition_contract(scene: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "camera",
        "canonical",
        "base_scene_id",
        "base_scene_fingerprint",
        "composition",
        "lighting_profile",
        "lineage",
        "license",
        "in_scene_lights",
    )
    for key in required:
        if key not in scene:
            raise ValueError(f"composition manifest 缺少字段 {key}")
    fixtures = scene["in_scene_lights"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("composition manifest 必须包含至少一个 in_scene_lights fixture")
    for fixture in fixtures:
        validate_fixture_contract(fixture)
    require_digest(scene["base_scene_fingerprint"], "base_scene_fingerprint")
    camera = scene["camera"]
    if not isinstance(camera, dict):
        raise ValueError("camera 必须是 mapping")
    require_vector(camera.get("location"), 3, "camera.location")
    require_vector(camera.get("rotation_euler"), 3, "camera.rotation_euler")
    require_vector(camera.get("target"), 3, "camera.target")
    if camera.get("coordinate_space") != "blender_world_meter":
        raise ValueError("camera.coordinate_space 必须是 blender_world_meter")
    if camera.get("strategy") != "generated_target_visible":
        raise ValueError("composition camera 必须由 generated_target_visible 策略生成")
    if int(camera.get("candidate_index", -1)) < 0:
        raise ValueError("camera.candidate_index 非法")
    if int(camera.get("coarse_candidate_index", -1)) < 0:
        raise ValueError("camera.coarse_candidate_index 非法")
    if int(camera.get("repair_index", -1)) < 0:
        raise ValueError("camera.repair_index 非法")
    if int(camera.get("target_visible_pixels", 0)) <= 0:
        raise ValueError("composition camera 中没有可见 target entity 区域")
    reference_pixels = int(camera.get("inserted_reference_pixels", 0))
    visible_pixels = int(scene.get("composition", {}).get("inserted_visible_pixels", 0))
    visible_ratio = float(camera.get("inserted_visible_ratio", float("nan")))
    if reference_pixels <= 0 or visible_pixels <= 0 or visible_pixels > reference_pixels:
        raise ValueError("composition camera 主体 reference/visible pixels 非法")
    if not np.isfinite(visible_ratio) or not 0.0 < visible_ratio <= 1.0:
        raise ValueError("camera.inserted_visible_ratio 非法")
    if not np.isclose(visible_ratio, visible_pixels / reference_pixels, rtol=1e-5, atol=1e-6):
        raise ValueError("camera.inserted_visible_ratio 与像素计数不一致")
    clearance = float(camera.get("clearance", float("nan")))
    foreground = float(camera.get("near_foreground_fraction", float("nan")))
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("camera.clearance 非法")
    if not np.isfinite(foreground) or not 0.0 <= foreground <= 1.0:
        raise ValueError("camera.near_foreground_fraction 非法")
    for key in ("shift_x", "shift_y"):
        if not np.isfinite(float(camera.get(key, float("nan")))):
            raise ValueError(f"camera.{key} 包含 NaN/Inf")
    ndc_bounds = camera.get("inserted_ndc_bounds")
    if not isinstance(ndc_bounds, dict):
        raise ValueError("camera.inserted_ndc_bounds 必须是 mapping")
    for key in ("min_x", "max_x", "min_y", "max_y", "width", "height", "center_x", "center_y", "min_depth"):
        if key not in ndc_bounds or not np.isfinite(float(ndc_bounds[key])):
            raise ValueError(f"camera.inserted_ndc_bounds.{key} 缺失或非有限值")
    if not (
        0.0 <= float(ndc_bounds["min_x"]) < float(ndc_bounds["max_x"]) <= 1.0
        and 0.0 <= float(ndc_bounds["min_y"]) < float(ndc_bounds["max_y"]) <= 1.0
        and float(ndc_bounds["min_depth"]) > 0.0
    ):
        raise ValueError("插入对象没有完整位于 composition camera 视锥内")
    target_ndc = camera.get("target_entity_ndc")
    if not isinstance(target_ndc, list) or not target_ndc:
        raise ValueError("camera.target_entity_ndc 不能为空")
    for index, point in enumerate(target_ndc):
        projected = require_vector(point, 3, f"camera.target_entity_ndc[{index}]")
        if not (0.0 <= projected[0] <= 1.0 and 0.0 <= projected[1] <= 1.0 and projected[2] > 0.0):
            raise ValueError("target entity reference point 不在 composition camera 视锥内")
    canonical = scene["canonical"]
    if not isinstance(canonical, dict):
        raise ValueError("canonical 必须是 mapping")
    require_vector(canonical.get("origin"), 3, "canonical.origin")
    if not np.allclose(camera["target"], canonical["origin"], atol=1e-6):
        raise ValueError("camera.target 与 canonical.origin 不一致")
    asset_size = float(canonical.get("asset_size", 0))
    if not np.isfinite(asset_size) or asset_size <= 0:
        raise ValueError("canonical.asset_size 必须为正数")
    if canonical.get("position_axes") != "x=right,y=camera-forward,z=up":
        raise ValueError("canonical.position_axes 与 TokenLight reader 约定不一致")
    composition = scene["composition"]
    if composition.get("relation") not in {"place_on", "replace"}:
        raise ValueError("composition.relation 非法")
    if not composition.get("target_entity_id"):
        raise ValueError("composition.target_entity_id 为空")
    require_vector(composition.get("object_transform_world"), 16, "composition.object_transform_world")
    if int(composition.get("inserted_visible_pixels", 0)) <= 0:
        raise ValueError("composition 插入对象没有可见像素")
    initial_scale = np.asarray(
        require_vector(
            composition.get("initial_asset_scale"), 3, "composition.initial_asset_scale"
        ),
        dtype=np.float64,
    )
    final_scale = np.asarray(
        require_vector(composition.get("asset_scale"), 3, "composition.asset_scale"),
        dtype=np.float64,
    )
    if np.any(initial_scale <= 0.0) or np.any(final_scale <= 0.0):
        raise ValueError("composition asset scale 必须为正数")
    collision_pairs = composition.get("collision_pairs")
    if collision_pairs != []:
        raise ValueError("通过验收的 composition.collision_pairs 必须为空")
    collision_resolution = composition.get("collision_resolution")
    if not isinstance(collision_resolution, dict):
        raise ValueError("composition.collision_resolution 必须是 mapping")
    strategy = collision_resolution.get("strategy")
    if strategy not in {"none", "uniform_shrink"}:
        raise ValueError("composition.collision_resolution.strategy 非法")
    scale_ratio = float(collision_resolution.get("scale_ratio", float("nan")))
    attempts = int(collision_resolution.get("attempts", 0))
    if not np.isfinite(scale_ratio) or not 0.0 < scale_ratio <= 1.0 or attempts < 1:
        raise ValueError("composition collision scale_ratio/attempts 非法")
    if strategy == "none" and (attempts != 1 or not np.isclose(scale_ratio, 1.0)):
        raise ValueError("未缩放的 composition 必须记录 ratio=1 且 attempts=1")
    if strategy == "uniform_shrink" and not scale_ratio < 1.0:
        raise ValueError("uniform_shrink 必须记录小于 1 的 scale_ratio")
    if not np.allclose(final_scale, initial_scale * scale_ratio, rtol=1e-6, atol=1e-8):
        raise ValueError("composition 最终 asset_scale 与 uniform scale_ratio 不一致")
    lineage = scene["lineage"]
    for key in (
        "annotation_digest",
        "object_digest",
        "scene_digest",
        "prepared_geometry_digest",
        "base_scene_source_digest",
        "render_contract_digest",
        "render_job_digest",
        "config_digest",
        "generator_version",
    ):
        if key not in lineage:
            raise ValueError(f"lineage 缺少字段 {key}")
        if key.endswith("digest"):
            require_digest(lineage[key], f"lineage.{key}")
    license_record = scene["license"]
    if license_record.get("decision") != "allowed":
        raise ValueError("license decision 不是 allowed")
    if not license_record.get("policy_version") or license_record["policy_version"] == "unconfigured":
        raise ValueError("license policy_version 未配置")


def require_camera_position(scene: dict[str, Any], light: dict[str, Any], label: str) -> None:
    """Reject manifests whose training coordinates are missing or inconsistent.

    Training consumes ``camera_position`` = ``[right, down, forward]`` in meters
    from the camera center.  The value is checked against the canonical
    ``position`` so a manifest can never ship two disagreeing frames.
    """
    value = require_vector(light.get("camera_position"), 3, f"{label}.camera_position")
    frame = coordinates.CameraFrame.from_row(scene)
    expected = frame.canonical_to_camera(light["position"])
    if not all(abs(a - b) <= 1e-5 for a, b in zip(value, expected)):
        raise ValueError(
            f"{label}.camera_position disagrees with position: {value} vs {expected}"
        )


def validate_fixture_contract(fixture: Any) -> None:
    if not isinstance(fixture, dict):
        raise ValueError("in_scene_lights fixture 必须是 mapping")
    source = fixture.get("fixture_source")
    if source not in {"scene_native", "procedural_fallback"}:
        raise ValueError(f"fixture_source 非法: {source}")
    entity_id = fixture.get("fixture_entity_id")
    if source == "scene_native" and not entity_id:
        raise ValueError("scene_native fixture 缺少 fixture_entity_id")
    if source == "procedural_fallback" and entity_id is not None:
        raise ValueError("procedural_fallback fixture 不得声明 scene entity")
    require_vector(fixture.get("position"), 3, "fixture.position")
    require_vector(fixture.get("renderer_position"), 3, "fixture.renderer_position")
    screen_fraction = float(fixture.get("screen_fraction", float("nan")))
    if not np.isfinite(screen_fraction) or not 0.0 < screen_fraction <= 1.0:
        raise ValueError("fixture.screen_fraction 非法")


def require_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} 必须包含 {length} 个数")
    result = [float(item) for item in value]
    if not np.isfinite(result).all():
        raise ValueError(f"{label} 包含 NaN/Inf")
    return result


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} 不是规范 sha256 digest")


def read_linear(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path).astype(np.float32)
    else:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV 无法读取 {path}")
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3].astype(np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError(f"无效线性图像 {path}: shape={image.shape}, finite={np.isfinite(image).all()}")
    return image


def require_shape(image: np.ndarray, reference: np.ndarray, name: str) -> None:
    if image.shape != reference.shape:
        raise ValueError(f"component shape 不一致 {name}: {image.shape} vs {reference.shape}")


def read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        import cv2

        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"OpenCV 无法读取 fixture mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> dict:
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
