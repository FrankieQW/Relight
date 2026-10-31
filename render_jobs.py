from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any

from . import __version__
from .config import ProjectConfig
from .io_utils import (
    dump_json_atomic,
    dump_jsonl_atomic,
    load_json,
    sha256_file,
    stable_digest,
    utc_now,
)
from .schemas import (
    AnnotationDocument,
    ObjectDocument,
    PreparedGeometryDocument,
    RenderJob,
    RenderJobDocument,
    SceneDocument,
)


_RENDER_JOB_CONTRACT_VERSION = "4"


def build_render_jobs(
    config: ProjectConfig,
    *,
    annotations_path: Path | None = None,
    output_path: Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    annotation_path = annotations_path or config.path("annotation_output")
    object_path = config.path("object_output")
    scene_path = config.path("scene_output")
    geometry_path = config.path("prepared_geometry_output")
    annotations = AnnotationDocument.model_validate(load_json(annotation_path)).model_dump(mode="json")
    objects = ObjectDocument.model_validate(load_json(object_path)).model_dump(mode="json")
    scenes = SceneDocument.model_validate(load_json(scene_path)).model_dump(mode="json")
    prepared = PreparedGeometryDocument.model_validate(load_json(geometry_path)).model_dump(mode="json")

    current_digests = {
        "annotation": sha256_file(annotation_path),
        "object": sha256_file(object_path),
        "scene": sha256_file(scene_path),
        "prepared_geometry": sha256_file(geometry_path),
    }
    if annotations["object_digest"] != current_digests["object"]:
        raise ValueError("annotation_construction.json references a stale object.json")
    if annotations["scene_digest"] != current_digests["scene"]:
        raise ValueError("annotation_construction.json references a stale scene.json")
    if prepared["object_digest"] != current_digests["object"]:
        raise ValueError("prepared geometry references a stale object.json")

    m4 = config.section("m4")
    selected_seed = int(seed if seed is not None else config.section("project").get("seed", 0))
    relation_priority = list(m4.get("relation_priority", ["replace", "place_on"]))
    if sorted(relation_priority) != ["place_on", "replace"]:
        raise ValueError("m4.relation_priority must contain place_on and replace exactly once")
    yaw_choices = [float(value) for value in m4.get("yaw_degrees", [0, 90, 180, 270])]
    if not yaw_choices or any(not math.isfinite(value) for value in yaw_choices):
        raise ValueError("m4.yaw_degrees must contain finite values")
    fixture_categories = {str(value) for value in m4.get("fixture_categories", [])}
    camera_strategy = str(m4.get("camera_strategy", "generated_target_visible"))
    if camera_strategy != "generated_target_visible":
        raise ValueError("m4.camera_strategy must be generated_target_visible")
    replacement_fill = float(m4.get("replacement_fill_ratio", 0.9))
    if not math.isfinite(replacement_fill) or not 0 < replacement_fill <= 1:
        raise ValueError("m4.replacement_fill_ratio must be in (0, 1]")

    object_by_uid = {row["uid"]: row for row in objects["objects"]}
    geometry_by_uid = {row["object_uid"]: row for row in prepared["geometries"]}
    targets = annotations["targets_by_object_category"]
    confidence_threshold = float(
        config.section("annotation").get("confidence_review_threshold", 0.8)
    )
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("annotation.confidence_review_threshold must be finite and in [0, 1]")
    rule_index = {
        (rule["object_category"], rule["scene_category"]): rule
        for rule in annotations["class_rules"]
    }
    jobs: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    max_jobs_value = m4.get("max_render_jobs")
    max_jobs: int | None = None
    if max_jobs_value is not None:
        if isinstance(max_jobs_value, bool):
            raise ValueError("m4.max_render_jobs must be a positive integer or null")
        max_jobs = int(max_jobs_value)
        if max_jobs <= 0:
            raise ValueError("m4.max_render_jobs must be a positive integer or null")
    render_contract_digest = stable_digest(
        {
            "contract_version": _RENDER_JOB_CONTRACT_VERSION,
            "generator_version": __version__,
            "render": m4.get("render", {}),
            "fixture": m4.get("fixture", {}),
        }
    )
    blend_digests: dict[Path, str] = {}

    for geometry in sorted(prepared["geometries"], key=lambda row: row["object_uid"]):
        jobs_before_object = len(jobs)
        uid = geometry["object_uid"]
        object_row = object_by_uid.get(uid)
        if object_row is None:
            rejects.append({"object_uid": uid, "reason": "missing_object_row"})
            continue
        category_targets = targets.get(geometry["object_category"], {})
        allowed = {
            "place_on": set(category_targets.get("place_on_entity_ids", [])),
            "replace": set(category_targets.get("replace_entity_ids", [])),
        }
        for scene in sorted(scenes["scenes"], key=lambda row: row["scene_id"]):
            blend_path = (config.root / scene["normalized_blend"]).resolve()
            if not blend_path.is_file():
                rejects.append(
                    {
                        "object_uid": uid,
                        "base_scene_id": scene["scene_id"],
                        "reason": f"missing_base_scene_blend:{blend_path}",
                    }
                )
                continue
            if blend_path not in blend_digests:
                blend_digests[blend_path] = sha256_file(blend_path)
            base_scene_blend_digest = blend_digests[blend_path]
            entities = {row["entity_id"]: row for row in scene["entities"]}
            candidates = {
                relation: [
                    entities[entity_id]
                    for entity_id in sorted(ids)
                    if entity_id in entities
                    and _rule_allows_target(
                        rule_index,
                        geometry["object_category"],
                        entities[entity_id]["category"],
                        relation,
                        confidence_threshold,
                    )
                ]
                for relation, ids in allowed.items()
            }
            relation = next((item for item in relation_priority if candidates[item]), None)
            if relation is None:
                continue
            rng = random.Random(_seed_for(selected_seed, uid, scene["scene_id"]))
            target = candidates[relation][rng.randrange(len(candidates[relation]))]
            dimensions = _three_positive(target.get("obb_world", {}).get("dimensions"), "target dimensions")
            center = _three(target.get("obb_world", {}).get("center"), "target center")
            if relation == "replace":
                desired_dimensions = [value * replacement_fill for value in dimensions]
                bottom_center = [center[0], center[1], center[2] - dimensions[2] * 0.5]
            else:
                desired_dimensions = _three_positive(
                    geometry["target_dimensions"], "prepared geometry target dimensions"
                )
                bottom_center = [center[0], center[1], center[2] + dimensions[2] * 0.5]
            yaw = yaw_choices[rng.randrange(len(yaw_choices))]
            scene_camera = _scene_override(config, scene["scene_id"])
            annotation_id = stable_digest(
                {
                    "annotation": current_digests["annotation"],
                    "category": geometry["object_category"],
                    "relation": relation,
                    "entity_id": target["entity_id"],
                }
            )
            job_payload = {
                "schema_version": str(config.section("project").get("schema_version", "1.0")),
                "seed": _seed_for(selected_seed, uid, scene["scene_id"], target["entity_id"]),
                "annotation_id": annotation_id,
                "object_uid": uid,
                "object_category": geometry["object_category"],
                "object_asset_path": geometry["asset_path"],
                "prepared_geometry": geometry,
                "base_scene_id": scene["scene_id"],
                "base_scene_blend": scene["normalized_blend"],
                "base_scene_digest": base_scene_blend_digest,
                "target": {
                    "relation": relation,
                    "entity_id": target["entity_id"],
                    "node_ids": target["node_ids"],
                    "category": target["category"],
                    "center_world": center,
                    "dimensions_world": dimensions,
                    "bottom_center_world": bottom_center,
                    "desired_dimensions": desired_dimensions,
                    "yaw_degrees": yaw,
                },
                "camera": {
                    "strategy": camera_strategy,
                    "focal_length": float(m4.get("camera_focal_length", 50.0)),
                    "candidate_count": int(scene_camera.get("candidate_count", m4.get("camera_candidate_count", 24))),
                    "azimuth_degrees": list(scene_camera.get("azimuth_degrees", m4.get("camera_azimuth_degrees", [0, 45, 90, 135, 180, 225, 270, 315]))),
                    "elevation_degrees": list(scene_camera.get("elevation_degrees", m4.get("camera_elevation_degrees", [15, 25, 35, 45]))),
                    "subject_fill_range": list(scene_camera.get("subject_fill_range", m4.get("camera_subject_fill_range", [0.25, 0.60]))),
                    "shift_x_range": list(scene_camera.get("shift_x_range", m4.get("camera_shift_x_range", [-0.18, 0.18]))),
                    "shift_y_range": list(scene_camera.get("shift_y_range", m4.get("camera_shift_y_range", [-0.15, 0.15]))),
                    "ndc_x_range": list(scene_camera.get("ndc_x_range", m4.get("camera_ndc_x_range", [0.15, 0.85]))),
                    "ndc_y_range": list(scene_camera.get("ndc_y_range", m4.get("camera_ndc_y_range", [0.15, 0.85]))),
                    "edge_margin": float(scene_camera.get("edge_margin", m4.get("camera_edge_margin", 0.02))),
                    "target_minimum_visible_pixels": int(scene_camera.get("target_minimum_visible_pixels", m4.get("camera_target_minimum_visible_pixels", 64))),
                    "repair_attempts_per_candidate": int(scene_camera.get("repair_attempts_per_candidate", m4.get("camera_repair_attempts_per_candidate", 6))),
                    "scored_candidate_limit": int(scene_camera.get("scored_candidate_limit", m4.get("camera_scored_candidate_limit", 6))),
                    "minimum_visible_ratio": float(scene_camera.get("minimum_visible_ratio", m4.get("camera_minimum_visible_ratio", 0.70))),
                    "minimum_clearance": float(scene_camera.get("minimum_clearance", m4.get("camera_minimum_clearance", 0.05))),
                    "clearance_subject_ratio": float(scene_camera.get("clearance_subject_ratio", m4.get("camera_clearance_subject_ratio", 0.10))),
                    "foreground_grid_size": int(scene_camera.get("foreground_grid_size", m4.get("camera_foreground_grid_size", 12))),
                    "maximum_foreground_fraction": float(scene_camera.get("maximum_foreground_fraction", m4.get("camera_maximum_foreground_fraction", 0.35))),
                    "foreground_depth_margin_ratio": float(scene_camera.get("foreground_depth_margin_ratio", m4.get("camera_foreground_depth_margin_ratio", 0.20))),
                },
                "lighting_profile": {
                    "name": str(m4.get("lighting_profile", "tokenlight_linear_v1")),
                    "native_analytic": False,
                    "native_emissive": False,
                    "component_world_strength": 0.0,
                },
                "fixture_candidate_entity_ids": sorted(
                    entity["entity_id"]
                    for entity in scene["entities"]
                    if entity["category"] in fixture_categories
                ),
                "lineage": {
                    "annotation_digest": current_digests["annotation"],
                    "object_digest": current_digests["object"],
                    "scene_digest": current_digests["scene"],
                    "prepared_geometry_digest": current_digests["prepared_geometry"],
                    "base_scene_source_digest": scene["source_digest"],
                    "render_contract_digest": render_contract_digest,
                    "config_digest": config.digest,
                    "generator_version": __version__,
                },
                "license": {
                    "object": geometry["license"],
                    "base_scene": m4.get("base_scene_license", {}),
                    "policy_version": str(m4.get("license_policy_version", "unconfigured")),
                    "decision": "allowed"
                    if geometry["license"].get("decision") == "allowed"
                    and m4.get("base_scene_license", {}).get("decision") == "allowed"
                    else "unverified",
                },
            }
            _validate_camera_contract(job_payload["camera"])
            render_job_digest = stable_digest(_job_identity(job_payload))
            job_payload["lineage"]["render_job_digest"] = render_job_digest
            job_payload["job_id"] = _job_id(render_job_digest)
            jobs.append(RenderJob.model_validate(job_payload).model_dump(mode="json"))
            if max_jobs is not None and len(jobs) >= int(max_jobs):
                break
        if len(jobs) == jobs_before_object:
            rejects.append(
                {
                    "object_uid": uid,
                    "object_category": geometry["object_category"],
                    "reason": "no_eligible_high_confidence_target",
                }
            )
        if max_jobs is not None and len(jobs) >= int(max_jobs):
            break

    jobs.sort(key=lambda row: row["job_id"])
    document = RenderJobDocument(
        schema_version=str(config.section("project").get("schema_version", "1.0")),
        generated_at=utc_now(),
        generator_version=__version__,
        config_digest=config.digest,
        annotation_digest=current_digests["annotation"],
        object_digest=current_digests["object"],
        scene_digest=current_digests["scene"],
        prepared_geometry_digest=current_digests["prepared_geometry"],
        jobs=jobs,
        rejects=rejects,
        stats={
            "prepared_objects": len(geometry_by_uid),
            "scenes": len(scenes["scenes"]),
            "jobs": len(jobs),
            "place_on": sum(row["target"]["relation"] == "place_on" for row in jobs),
            "replace": sum(row["target"]["relation"] == "replace" for row in jobs),
            "rejects": len(rejects),
        },
    ).model_dump(mode="json")
    selected_output = output_path or config.path("render_jobs_output")
    dump_jsonl_atomic(selected_output, document["jobs"])
    dump_json_atomic(config.path("render_jobs_summary"), document)
    dump_jsonl_atomic(config.path("render_job_rejects"), document["rejects"])
    if not document["jobs"]:
        raise RuntimeError(
            "render-job generation produced zero jobs; inspect render_job_rejects and annotation targets"
        )
    return document


def _seed_for(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _job_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "seed",
            "annotation_id",
            "object_uid",
            "object_category",
            "object_asset_path",
            "prepared_geometry",
            "base_scene_id",
            "base_scene_blend",
            "base_scene_digest",
            "target",
            "camera",
            "lighting_profile",
            "fixture_candidate_entity_ids",
            "license",
        )
    } | {"render_contract_digest": payload["lineage"]["render_contract_digest"]}


def _job_id(render_job_digest: str) -> str:
    return "composition_" + render_job_digest.removeprefix("sha256:")[:20]


def _three(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _three_positive(value: Any, label: str) -> list[float]:
    result = _three(value, label)
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} must be positive")
    return result


def _scene_override(config: ProjectConfig, scene_id: str) -> dict[str, Any]:
    overrides = config.section("m4").get("scene_render_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("m4.scene_render_overrides must be a mapping")
    value = overrides.get(scene_id, {})
    if not isinstance(value, dict):
        raise TypeError(f"scene render override must be a mapping: {scene_id}")
    return value


def _validate_camera_contract(camera: dict[str, Any]) -> None:
    scalar_fields = (
        "focal_length",
        "edge_margin",
        "minimum_visible_ratio",
        "minimum_clearance",
        "clearance_subject_ratio",
        "maximum_foreground_fraction",
        "foreground_depth_margin_ratio",
    )
    sequence_fields = (
        "azimuth_degrees",
        "elevation_degrees",
        "subject_fill_range",
        "shift_x_range",
        "shift_y_range",
        "ndc_x_range",
        "ndc_y_range",
    )
    for field in scalar_fields:
        value = float(camera[field])
        if not math.isfinite(value):
            raise ValueError(f"camera.{field} must be finite")
    for field in sequence_fields:
        values = camera[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"camera.{field} must be a non-empty list")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"camera.{field} must contain finite values")
    if int(camera["candidate_count"]) <= 0:
        raise ValueError("camera.candidate_count must be positive")
    if int(camera["target_minimum_visible_pixels"]) <= 0:
        raise ValueError("camera.target_minimum_visible_pixels must be positive")
    if int(camera["repair_attempts_per_candidate"]) <= 0:
        raise ValueError("camera.repair_attempts_per_candidate must be positive")
    if int(camera["scored_candidate_limit"]) <= 0:
        raise ValueError("camera.scored_candidate_limit must be positive")
    if int(camera["foreground_grid_size"]) < 2:
        raise ValueError("camera.foreground_grid_size must be at least 2")
    if not 0.0 < float(camera["minimum_visible_ratio"]) <= 1.0:
        raise ValueError("camera.minimum_visible_ratio must be in (0, 1]")
    if float(camera["minimum_clearance"]) < 0.0 or float(camera["clearance_subject_ratio"]) < 0.0:
        raise ValueError("camera clearance values must be non-negative")
    if not 0.0 <= float(camera["maximum_foreground_fraction"]) <= 1.0:
        raise ValueError("camera.maximum_foreground_fraction must be in [0, 1]")
    if float(camera["foreground_depth_margin_ratio"]) < 0.0:
        raise ValueError("camera.foreground_depth_margin_ratio must be non-negative")


def _rule_allows_target(
    rules: dict[tuple[str, str], dict[str, Any]],
    object_category: str,
    scene_category: str,
    relation: str,
    confidence_threshold: float,
) -> bool:
    rule = rules.get((object_category, scene_category))
    if rule is None:
        return False
    confidence = float(rule["confidence"])
    if not math.isfinite(confidence) or confidence < confidence_threshold:
        return False
    field = "can_replace" if relation == "replace" else "can_place_on"
    return bool(rule[field])
