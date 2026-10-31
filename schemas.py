from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectAsset(StrictModel):
    uid: str
    primary_category: str
    primary_category_normalized: str
    categories: list[str]
    categories_normalized: list[str]
    inventory_path: str
    canonical_path: str
    shard: str
    license_raw: str | None
    license: str | None
    source_uri: str | None
    embed_url: str | None
    name: str | None
    description: str | None
    tags: list[str]
    author: dict[str, Any]
    is_age_restricted: bool | None
    is_downloadable: bool | None
    glb_stats: dict[str, Any] | None
    thumbnails: list[dict[str, Any]]
    metadata_status: str


class ObjectDocument(StrictModel):
    schema_version: str
    generated_at: str
    generator_version: str
    config_digest: str
    inventory_mode: str
    object_root_env: str
    source_digests: dict[str, str]
    stats: dict[str, Any]
    objects: list[ObjectAsset]


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


class ConstructionDecision(StrictModel):
    can_place_on: bool = Field(
        description="Whether an instance of the object category can plausibly rest on top of the scene category."
    )
    can_replace: bool = Field(
        description="Whether the object category can plausibly replace an instance of the scene category."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=240)


class ClassRule(StrictModel):
    object_category: str
    scene_category: str
    can_place_on: bool
    can_replace: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    cache_key: str


class AnnotationModelInfo(StrictModel):
    name: str
    base_url: str
    prompt_version: str
    temperature: float
    thinking: bool


class AnnotationDocument(StrictModel):
    schema_version: str
    generated_at: str
    generator_version: str
    config_digest: str
    object_digest: str
    scene_digest: str
    model: AnnotationModelInfo
    class_rules: list[ClassRule]
    targets_by_object_category: dict[str, dict[str, list[str]]]
    object_index: dict[str, str]
    unresolved_pairs: list[dict[str, Any]]
    stats: dict[str, Any]


class PreparedGeometry(StrictModel):
    object_uid: str
    object_category: str
    asset_path: str
    asset_digest: str
    target_dimensions: list[float] = Field(min_length=3, max_length=3)
    up_axis: str
    front_axis: str
    contact_axis: str
    fit_mode: str
    config_source: str
    license: dict[str, Any]


class PreparedGeometryDocument(StrictModel):
    schema_version: str
    generated_at: str
    generator_version: str
    config_digest: str
    object_digest: str
    geometry_config_digests: dict[str, str]
    object_root_env: str
    geometries: list[PreparedGeometry]
    quarantine: list[dict[str, Any]]
    stats: dict[str, Any]


class CompositionTarget(StrictModel):
    relation: str
    entity_id: str
    node_ids: list[str] = Field(min_length=1)
    category: str
    center_world: list[float] = Field(min_length=3, max_length=3)
    dimensions_world: list[float] = Field(min_length=3, max_length=3)
    bottom_center_world: list[float] = Field(min_length=3, max_length=3)
    desired_dimensions: list[float] = Field(min_length=3, max_length=3)
    yaw_degrees: float


class RenderJob(StrictModel):
    schema_version: str
    job_id: str
    seed: int
    annotation_id: str
    object_uid: str
    object_category: str
    object_asset_path: str
    prepared_geometry: PreparedGeometry
    base_scene_id: str
    base_scene_blend: str
    base_scene_digest: str
    target: CompositionTarget
    camera: dict[str, Any]
    lighting_profile: dict[str, Any]
    fixture_candidate_entity_ids: list[str]
    lineage: dict[str, Any]
    license: dict[str, Any]


class RenderJobDocument(StrictModel):
    schema_version: str
    generated_at: str
    generator_version: str
    config_digest: str
    annotation_digest: str
    object_digest: str
    scene_digest: str
    prepared_geometry_digest: str
    jobs: list[RenderJob]
    rejects: list[dict[str, Any]]
    stats: dict[str, Any]
