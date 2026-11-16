"""Entity-conditioned camera targets: pick the object a sample is *about*.

The scene-only pipeline has no object prompt, so a camera pose can end up
framing nothing but a wall corner: two large planes fill the frame, no single
node dominates, and the depth spread is large - which is exactly what the
free-space gates reward.  The fix is to add the missing ingredient: a named
target entity.  A pose is then scored on *what it looks at* instead of on how
much surface it happens to fill.

Everything here is plain NumPy over ``scene.json`` (no ``bpy``, no pydantic, no
yaml).  Node AABBs are a broad phase; the exact triangle test happens later in
``blender/render_scene.py::_validate_camera_pose``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_IGNORE_CATEGORIES = (
    "building",
    "ceiling",
    "floor",
    "ground",
    "road",
    "sidewalk",
    "sky",
    "wall",
)

DEFAULT_IGNORE_AS_OCCLUDER = (
    "curtain",
    "foliage",
    "glass",
    "railing",
    "sky",
    "window",
)


@dataclass(frozen=True)
class TargetSpec:
    """Thresholds for target selection and entity-conditioned pose sampling.

    Every value is exposed through ``m4.camera.target`` in ``configs/default.yaml``
    so the plan can be tuned without touching code.
    """

    enabled: bool = True

    # -- selection ---------------------------------------------------------
    min_diagonal: float = 0.15  # metres; smaller entities are not worth framing
    max_targets_per_scene: int = 4
    duplicate_iou: float = 0.80  # overlapping candidates are counted once
    ignore_categories: tuple[str, ...] = DEFAULT_IGNORE_CATEGORIES

    # -- candidate generation ---------------------------------------------
    radius_scale: float = 2.6  # r = radius_scale * half diagonal
    min_radius: float = 0.60
    max_radius: float = 6.00
    radius_jitter: tuple[float, float] = (0.85, 1.50)
    pitch_degrees: tuple[float, float] = (-10.0, 45.0)
    azimuth_samples: int = 12
    azimuth_jitter_degrees: float = 15.0
    look_jitter_degrees: float = 0.0
    attempts_per_target: int = 200

    # -- gates -------------------------------------------------------------
    surface_samples: int = 64
    min_target_visible_ratio: float = 0.40
    min_target_screen_ratio: float = 0.02
    max_target_screen_ratio: float = 0.75
    in_frame_margin: float = 0.90
    contact_tolerance: float = 0.02

    # -- occlusion handling ------------------------------------------------
    ignore_as_occluder_categories: tuple[str, ...] = DEFAULT_IGNORE_AS_OCCLUDER
    thin_occluder_thickness: float = 0.01

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "min_diagonal": float(self.min_diagonal),
            "max_targets_per_scene": int(self.max_targets_per_scene),
            "duplicate_iou": float(self.duplicate_iou),
            "ignore_categories": list(self.ignore_categories),
            "radius_scale": float(self.radius_scale),
            "min_radius": float(self.min_radius),
            "max_radius": float(self.max_radius),
            "radius_jitter": [float(value) for value in self.radius_jitter],
            "pitch_degrees": [float(value) for value in self.pitch_degrees],
            "azimuth_samples": int(self.azimuth_samples),
            "azimuth_jitter_degrees": float(self.azimuth_jitter_degrees),
            "look_jitter_degrees": float(self.look_jitter_degrees),
            "attempts_per_target": int(self.attempts_per_target),
            "surface_samples": int(self.surface_samples),
            "min_target_visible_ratio": float(self.min_target_visible_ratio),
            "min_target_screen_ratio": float(self.min_target_screen_ratio),
            "max_target_screen_ratio": float(self.max_target_screen_ratio),
            "in_frame_margin": float(self.in_frame_margin),
            "contact_tolerance": float(self.contact_tolerance),
            "ignore_as_occluder_categories": list(self.ignore_as_occluder_categories),
            "thin_occluder_thickness": float(self.thin_occluder_thickness),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TargetSpec":
        """Build a spec from a YAML mapping, mirroring ``CameraSpec`` parsing."""
        source: Mapping[str, Any] = value or {}
        defaults = cls()
        return cls(
            enabled=bool(source.get("enabled", defaults.enabled)),
            min_diagonal=float(source.get("min_diagonal", defaults.min_diagonal)),
            max_targets_per_scene=int(
                source.get("max_targets_per_scene", defaults.max_targets_per_scene)
            ),
            duplicate_iou=float(source.get("duplicate_iou", defaults.duplicate_iou)),
            ignore_categories=tuple(
                str(item)
                for item in source.get("ignore_categories", defaults.ignore_categories)
            ),
            radius_scale=float(source.get("radius_scale", defaults.radius_scale)),
            min_radius=float(source.get("min_radius", defaults.min_radius)),
            max_radius=float(source.get("max_radius", defaults.max_radius)),
            radius_jitter=tuple(
                float(item) for item in source.get("radius_jitter", defaults.radius_jitter)
            ),
            pitch_degrees=tuple(
                float(item) for item in source.get("pitch_degrees", defaults.pitch_degrees)
            ),
            azimuth_samples=int(source.get("azimuth_samples", defaults.azimuth_samples)),
            azimuth_jitter_degrees=float(
                source.get("azimuth_jitter_degrees", defaults.azimuth_jitter_degrees)
            ),
            look_jitter_degrees=float(
                source.get("look_jitter_degrees", defaults.look_jitter_degrees)
            ),
            attempts_per_target=int(
                source.get("attempts_per_target", defaults.attempts_per_target)
            ),
            surface_samples=int(source.get("surface_samples", defaults.surface_samples)),
            min_target_visible_ratio=float(
                source.get("min_target_visible_ratio", defaults.min_target_visible_ratio)
            ),
            min_target_screen_ratio=float(
                source.get("min_target_screen_ratio", defaults.min_target_screen_ratio)
            ),
            max_target_screen_ratio=float(
                source.get("max_target_screen_ratio", defaults.max_target_screen_ratio)
            ),
            in_frame_margin=float(source.get("in_frame_margin", defaults.in_frame_margin)),
            contact_tolerance=float(
                source.get("contact_tolerance", defaults.contact_tolerance)
            ),
            ignore_as_occluder_categories=tuple(
                str(item)
                for item in source.get(
                    "ignore_as_occluder_categories", defaults.ignore_as_occluder_categories
                )
            ),
            thin_occluder_thickness=float(
                source.get("thin_occluder_thickness", defaults.thin_occluder_thickness)
            ),
        )


@dataclass
class SceneTarget:
    """One framable entity plus the sample points used to reason about it."""

    entity_id: str
    label: str
    category: str
    node_ids: tuple[str, ...]
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    center: tuple[float, float, float]
    half_diagonal: float
    samples: np.ndarray = field(repr=False)
    triangle_count: int = 0

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return tuple(
            float(self.maximum[index] - self.minimum[index]) for index in range(3)
        )

    @property
    def natural_radius(self) -> float:
        return float(self.half_diagonal)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description (no sample points)."""
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "category": self.category,
            "node_ids": list(self.node_ids),
            "center": [float(value) for value in self.center],
            "minimum": [float(value) for value in self.minimum],
            "maximum": [float(value) for value in self.maximum],
            "dimensions": [float(value) for value in self.dimensions],
            "half_diagonal": float(self.half_diagonal),
            "triangle_count": int(self.triangle_count),
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def surface_samples(
    minimum: Sequence[float],
    maximum: Sequence[float],
    count: int,
    rng: random.Random,
) -> np.ndarray:
    """Area-weighted points on a box surface, nudged slightly inward.

    The inward nudge keeps the points strictly inside the *same* AABB as the
    real mesh, which matters for a broad-phase raycast: a point sitting exactly
    on the face can be reported as a hit at ``t == 0`` and be discarded.
    """
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    dims = np.maximum(high - low, 1e-9)
    count = max(1, int(count))
    face_areas = [
        dims[1] * dims[2],
        dims[1] * dims[2],
        dims[0] * dims[2],
        dims[0] * dims[2],
        dims[0] * dims[1],
        dims[0] * dims[1],
    ]
    faces = rng.choices(range(6), weights=face_areas, k=count)
    u = [rng.random() for _ in range(count)]
    v = [rng.random() for _ in range(count)]
    points = np.empty((count, 3), dtype=np.float64)
    for index, face in enumerate(faces):
        axis = face // 2
        points[index, axis] = high[axis] if face % 2 == 0 else low[axis]
        others = [item for item in range(3) if item != axis]
        points[index, others[0]] = low[others[0]] + u[index] * dims[others[0]]
        points[index, others[1]] = low[others[1]] + v[index] * dims[others[1]]
    center = (low + high) * 0.5
    inset = np.minimum(dims * 0.01, 0.005)
    return points + np.sign(center[None, :] - points) * inset[None, :]


def project_points(
    location: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    angle_x: float,
    angle_y: float,
    points: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole projection to normalized image coordinates.

    ``u`` and ``v`` span ``[-1, 1]`` across the full frame; ``depth`` is the
    signed distance along the camera's forward axis (negative = behind).
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(location, dtype=np.float64)[None, :]
    relative = array - origin
    forward_array = np.asarray(forward, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    depth = relative @ forward_array
    x = relative @ right_array
    y = relative @ up_array
    half_x = max(math.tan(angle_x * 0.5), 1e-9)
    half_y = max(math.tan(angle_y * 0.5), 1e-9)
    safe = np.maximum(depth, 1e-6)
    u = np.where(depth > 1e-6, x / (safe * half_x), np.inf)
    v = np.where(depth > 1e-6, y / (safe * half_y), np.inf)
    return u, v, depth


def screen_bbox_ratio(u: np.ndarray, v: np.ndarray) -> float:
    """Fraction of the frame covered by the bounding box of the projections."""
    if u.size == 0:
        return 0.0
    du = min(2.0, float(np.max(u) - np.min(u)))
    dv = min(2.0, float(np.max(v) - np.min(v)))
    return float(max(0.0, min(1.0, (du * 0.5) * (dv * 0.5))))


def estimated_screen_ratio(
    half_diagonal: float, radius: float, tan_half_x: float
) -> float:
    """Cheap a-priori estimate used to skip targets that can never fill the frame."""
    if radius <= 1e-6 or tan_half_x <= 1e-6:
        return 0.0
    width_fraction = half_diagonal / (radius * tan_half_x)
    return float(max(0.0, min(1.0, width_fraction * width_fraction)))


def radius_for(half_diagonal: float, spec: TargetSpec) -> float:
    """Camera distance for a target, before the local free-space cap."""
    raw = float(spec.radius_scale) * float(half_diagonal)
    return float(min(spec.max_radius, max(spec.min_radius, raw)))


def look_angles(direction: Sequence[float]) -> tuple[float, float]:
    """``(yaw, pitch)`` in degrees for a direction, matching ``camera_basis``."""
    x, y, z = (float(value) for value in direction)
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return 0.0, 0.0
    x, y, z = x / length, y / length, z / length
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    yaw = math.degrees(math.atan2(y, x))
    return yaw, pitch


def _intersection_over_union(
    left: tuple[Sequence[float], Sequence[float]],
    right: tuple[Sequence[float], Sequence[float]],
) -> float:
    (a_low, a_high), (b_low, b_high) = left, right
    overlap = 1.0
    for axis in range(3):
        low = max(float(a_low[axis]), float(b_low[axis]))
        high = min(float(a_high[axis]), float(b_high[axis]))
        overlap *= max(0.0, high - low)
    if overlap <= 0.0:
        return 0.0
    a_volume = 1.0
    b_volume = 1.0
    for axis in range(3):
        a_volume *= max(0.0, float(a_high[axis]) - float(a_low[axis]))
        b_volume *= max(0.0, float(b_high[axis]) - float(b_low[axis]))
    union = a_volume + b_volume - overlap
    return float(overlap / union) if union > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _node_bounds(scene: Mapping[str, Any]) -> dict[str, tuple[list[float], list[float], int]]:
    bounds: dict[str, tuple[list[float], list[float], int]] = {}
    for node in scene.get("nodes") or []:
        node_id = str(node.get("node_id", ""))
        world = node.get("aabb_world") or {}
        low, high = world.get("min"), world.get("max")
        if not node_id or not isinstance(low, list) or not isinstance(high, list):
            continue
        if len(low) != 3 or len(high) != 3:
            continue
        values = [float(item) for item in list(low) + list(high)]
        if not all(math.isfinite(item) for item in values):
            continue
        if any(values[index + 3] < values[index] for index in range(3)):
            continue
        bounds[node_id] = (
            values[:3],
            values[3:],
            int(node.get("triangle_count") or 0),
        )
    return bounds


def entity_box(
    entity: Mapping[str, Any],
    bounds: Mapping[str, tuple[list[float], list[float], int]],
) -> tuple[list[float], list[float], int] | None:
    """Union AABB of an entity's nodes; ``None`` when no node has usable bounds."""
    lows: list[list[float]] = []
    highs: list[list[float]] = []
    triangles = 0
    for node_id in entity.get("node_ids") or []:
        entry = bounds.get(str(node_id))
        if entry is None:
            continue
        low, high, count = entry
        lows.append(list(low))
        highs.append(list(high))
        triangles += count
    if not lows:
        return None
    minimum = [min(row[axis] for row in lows) for axis in range(3)]
    maximum = [max(row[axis] for row in highs) for axis in range(3)]
    return minimum, maximum, triangles


def select_targets(
    scene: Mapping[str, Any],
    spec: TargetSpec,
    *,
    seed: int = 0,
    tan_half_x: float = 0.36,
) -> list[SceneTarget]:
    """Rank the scene's entities and keep the ones worth framing.

    Selection is purely geometric (size, pixel budget, overlap) because the
    category classifier is not reliable enough to be a whitelist; ``raw_label``
    is carried along for provenance only.
    """
    bounds = _node_bounds(scene)
    if not bounds:
        return []
    ignored = {str(item).lower() for item in spec.ignore_categories}
    candidates: list[tuple[float, SceneTarget]] = []
    for entity in scene.get("entities") or []:
        category = str(entity.get("category", "")).lower()
        if category in ignored:
            continue
        box = entity_box(entity, bounds)
        if box is None:
            continue
        minimum, maximum, triangles = box
        if triangles <= 0:
            continue
        dimensions = [maximum[axis] - minimum[axis] for axis in range(3)]
        if min(dimensions) <= 0.0:
            continue
        half_diagonal = 0.5 * math.sqrt(sum(value * value for value in dimensions))
        if half_diagonal * 2.0 < float(spec.min_diagonal):
            continue
        radius = radius_for(half_diagonal, spec)
        estimate = estimated_screen_ratio(half_diagonal, radius, tan_half_x)
        if estimate < float(spec.min_target_screen_ratio):
            continue
        center = [
            (float(minimum[axis]) + float(maximum[axis])) * 0.5 for axis in range(3)
        ]
        candidates.append(
            (
                half_diagonal,
                SceneTarget(
                    entity_id=str(entity.get("entity_id", "")),
                    label=str(entity.get("raw_label", "")),
                    category=category,
                    node_ids=tuple(str(value) for value in entity.get("node_ids") or []),
                    minimum=tuple(float(value) for value in minimum),
                    maximum=tuple(float(value) for value in maximum),
                    center=tuple(float(value) for value in center),
                    half_diagonal=float(half_diagonal),
                    samples=np.empty((0, 3), dtype=np.float64),
                    triangle_count=int(triangles),
                ),
            )
        )
    # Largest first, deterministic tie-break so the same scene always yields the
    # same target list regardless of entity ordering in scene.json.
    candidates.sort(key=lambda row: (-row[0], row[1].entity_id))
    chosen: list[SceneTarget] = []
    for _size, target in candidates:
        if len(chosen) >= max(1, int(spec.max_targets_per_scene)):
            break
        box = (target.minimum, target.maximum)
        if any(
            _intersection_over_union(box, (other.minimum, other.maximum))
            > float(spec.duplicate_iou)
            for other in chosen
        ):
            continue
        chosen.append(target)
    rng = random.Random(int(seed) ^ 0x5EED_7A26)
    for target in chosen:
        target.samples = surface_samples(
            target.minimum, target.maximum, spec.surface_samples, rng
        )
    return chosen


def occluder_exclusions(
    scene: Mapping[str, Any], spec: TargetSpec
) -> frozenset[str]:
    """Node ids that must not be treated as solid occluders.

    Glass, curtains and foliage are transparent enough that an AABB test would
    wrongly declare the target invisible; very thin nodes (panes, railings) are
    treated the same way.
    """
    excluded: set[str] = set()
    categories = {str(item).lower() for item in spec.ignore_as_occluder_categories}
    for entity in scene.get("entities") or []:
        if str(entity.get("category", "")).lower() not in categories:
            continue
        excluded.update(str(value) for value in entity.get("node_ids") or [])
    if spec.thin_occluder_thickness > 0.0:
        for node_id, (low, high, _count) in _node_bounds(scene).items():
            thickness = min(float(high[axis]) - float(low[axis]) for axis in range(3))
            if thickness < float(spec.thin_occluder_thickness):
                excluded.add(node_id)
    return frozenset(excluded)


__all__ = [
    "DEFAULT_IGNORE_AS_OCCLUDER",
    "DEFAULT_IGNORE_CATEGORIES",
    "SceneTarget",
    "TargetSpec",
    "entity_box",
    "estimated_screen_ratio",
    "look_angles",
    "occluder_exclusions",
    "project_points",
    "radius_for",
    "screen_bbox_ratio",
    "select_targets",
    "surface_samples",
]
