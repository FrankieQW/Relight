"""Data contracts for the scene-only pipeline.

Only scenes, scene entities and the planned camera jobs survive: the object /
annotation / composition-era models were removed together with that pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SceneNode(StrictModel):
    node_id: str
    raw_import_name: str
    source_fbx_element_id: int | str | None
    object_type: str
    transform_world: list[float] = Field(min_length=16, max_length=16)
    aabb_world: dict[str, list[float]]
    obb_world: dict[str, Any]
    materials: list[str]
    triangle_count: int = Field(ge=0)
    vertex_count: int = Field(ge=0)


class SceneEntity(StrictModel):
    entity_id: str
    node_ids: list[str] = Field(min_length=1)
    raw_label: str
    category: str
    category_confidence: float = Field(ge=0.0, le=1.0)
    grouping_confidence: float = Field(ge=0.0, le=1.0)
    grouping_method: str
    replaceable: bool
    support_surface: bool
    obb_world: dict[str, Any]
    override_applied: bool | None = None


class SceneAsset(StrictModel):
    scene_id: str
    source_fbx: str
    source_digest: str
    normalized_blend: str
    units: str
    up_axis: str
    config_digest: str
    nodes: list[SceneNode]
    entities: list[SceneEntity]
    stats: dict[str, Any]


class SceneDocument(StrictModel):
    schema_version: str
    generated_at: str
    generator_version: str
    config_digest: str
    source_digests: dict[str, str]
    stats: dict[str, Any]
    scenes: list[SceneAsset]


class SceneCamera(StrictModel):
    strategy: str
    location: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    rotation_euler: list[float] = Field(min_length=3, max_length=3)
    focal_length: float
    sensor_width: float
    sensor_fit: str
    shift_x: float
    shift_y: float
    angle_x: float
    angle_y: float
    resolution: dict[str, int]
    intrinsics: dict[str, float | int]
    yaw_degrees: float
    pitch_degrees: float
    forward: list[float] = Field(min_length=3, max_length=3)
    up: list[float] = Field(min_length=3, max_length=3)
    right: list[float] = Field(min_length=3, max_length=3)
    camera_clearance: float
    minimum_hit_depth: float
    hit_fraction: float
    near_fraction: float
    depth_spread_ratio: float
    single_node_fraction: float
    score: float


class SceneCameraJob(StrictModel):
    schema_version: str
    job_id: str
    seed: int
    scene_id: str
    base_scene_blend: str
    base_scene_digest: str
    camera: SceneCamera
    canonical: dict[str, Any]
    camera_position_convention: dict[str, Any]
    lighting_profile: dict[str, Any]
    free_space_report: dict[str, Any]
    fixture_entity_ids: list[str] = Field(default_factory=list)
    license: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, Any]


class SceneCameraJobDocument(StrictModel):
    schema_version: str
    generated_at: str
    contract: str
    seed: int
    cameras_per_scene: int
    camera_spec: dict[str, Any]
    lighting_profile: dict[str, Any]
    jobs: list[SceneCameraJob]
    rejects: list[dict[str, Any]]
    stats: dict[str, Any]
