"""Scene-only camera planning: the hard half of "render the scene, not an object".

This module is the Blender-free feasibility prototype for the scene-only data
synthesis plan (see ``docs/scene_only_render_feasibility.md``).  When the object
is dropped, the semantic object/scene matching disappears and the only remaining
problem is choosing camera poses that

* stand in free space (the camera is never inside a mesh),
* have a collision-free view frustum (no geometry closer than the near plane,
  so no wall/table/chair is cut through by the camera),
* actually look at a meaningful amount of scene content instead of a flat
  close-up of one surface.

Implementation notes
--------------------
* Input is a plain ``scene.json`` mapping (the output of ``prepare_scenes``); no
  pydantic, no ``bpy``, no ``yaml``, so this module runs anywhere numpy does.
* Geometry uses the per-node world AABBs that ``scene_extract`` already
  produced.  That is a *broad phase*: it is conservative and cheap.  The exact
  triangle test in Blender reuses ``BVHTree`` the same way
  ``blender/render_scene.py::_validate_camera_pose`` does, so a pose that
  passes here is re-validated exactly before rendering.
* Everything is deterministic: candidate order and acceptance depend only on
  ``(seed, scene_id)``.
* The emitted camera block satisfies the existing manifest contract in
  ``coordinates.py`` (``canonical.origin == camera.target``, no-roll, Blender
  XYZ euler, pinhole intrinsics).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from typing import Any, Iterable, Sequence

import numpy as np


SCENE_CAMERA_CONTRACT_VERSION = "1"
CANONICAL_AXES = "x=right,y=camera-forward,z=up"  # must match coordinates.CANONICAL_AXES
CAMERA_AXES = "x=right,y=down,z=forward"

# Entities that can be stood on.  ``scene_extract`` already labels these, so the
# eye-height policy costs nothing extra.
FLOOR_CATEGORIES = frozenset(
    {"floor", "ground", "sidewalk", "road", "stair", "stairs", "staircase", "landing", "platform"}
)


@dataclass(frozen=True)
class CameraSpec:
    """Camera search contract for one scene-only sample."""

    focal_length: float = 50.0
    sensor_width: float = 36.0
    resolution: tuple[int, int] = (512, 512)

    # Pose sampling
    yaw_degrees: tuple[float, float] = (0.0, 360.0)
    pitch_degrees: tuple[float, float] = (-18.0, 12.0)
    height_policy: str = "floor_standing"  # or "free"
    eye_height: tuple[float, float] = (1.10, 1.80)
    free_height_band: tuple[float, float] = (0.05, 0.90)  # fraction of scene bbox z

    # Free-space / visibility gates
    camera_radius: float = 0.20  # metres of clearance around the camera centre
    near_clearance: float = 0.30  # nothing closer than this along any cast ray
    max_depth: float = 45.0
    ray_grid: int = 24  # ray_grid**2 rays across the image (final verification)
    search_ray_grid: int = 12  # cheaper grid used while searching
    min_hit_fraction: float = 0.90
    near_window: float = 1.00
    max_near_fraction: float = 0.25
    max_single_node_fraction: float = 0.90
    min_depth_spread_ratio: float = 1.35

    # Search budget
    attempts_per_camera: int = 32
    min_camera_spacing: float = 0.0

    def angles(self) -> tuple[float, float]:
        width, height = self.resolution
        angle_x = 2.0 * math.atan(self.sensor_width / (2.0 * self.focal_length))
        angle_y = 2.0 * math.atan(math.tan(angle_x * 0.5) * height / width)
        return angle_x, angle_y

    def as_dict(self) -> dict[str, Any]:
        angle_x, angle_y = self.angles()
        return {
            "focal_length": float(self.focal_length),
            "sensor_width": float(self.sensor_width),
            "resolution": {"width": int(self.resolution[0]), "height": int(self.resolution[1])},
            "yaw_degrees": [float(value) for value in self.yaw_degrees],
            "pitch_degrees": [float(value) for value in self.pitch_degrees],
            "height_policy": self.height_policy,
            "eye_height": [float(value) for value in self.eye_height],
            "free_height_band": [float(value) for value in self.free_height_band],
            "camera_radius": float(self.camera_radius),
            "near_clearance": float(self.near_clearance),
            "max_depth": float(self.max_depth),
            "ray_grid": int(self.ray_grid),
            "search_ray_grid": int(self.search_ray_grid),
            "min_hit_fraction": float(self.min_hit_fraction),
            "near_window": float(self.near_window),
            "max_near_fraction": float(self.max_near_fraction),
            "max_single_node_fraction": float(self.max_single_node_fraction),
            "min_depth_spread_ratio": float(self.min_depth_spread_ratio),
            "attempts_per_camera": int(self.attempts_per_camera),
            "min_camera_spacing": float(self.min_camera_spacing),
            "angle_x": angle_x,
            "angle_y": angle_y,
        }


# --------------------------------------------------------------------------- #
# Free space
# --------------------------------------------------------------------------- #
class FreeSpace:
    """Axis-aligned occlusion volume built from ``scene.json`` node bounds."""

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, node_ids: Sequence[str]) -> None:
        if mins.shape != maxs.shape or mins.ndim != 2 or mins.shape[1] != 3:
            raise ValueError("free-space bounds must be (N, 3) arrays")
        if len(node_ids) != len(mins):
            raise ValueError("one node id is required per bounding box")
        self.mins = np.asarray(mins, dtype=np.float64)
        self.maxs = np.asarray(maxs, dtype=np.float64)
        self.node_ids = list(node_ids)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_scene(cls, scene: dict[str, Any]) -> "FreeSpace":
        mins: list[list[float]] = []
        maxs: list[list[float]] = []
        node_ids: list[str] = []
        for node in scene.get("nodes", []):
            bounds = node.get("aabb_world") or {}
            low, high = bounds.get("min"), bounds.get("max")
            if not isinstance(low, list) or not isinstance(high, list):
                continue
            if len(low) != 3 or len(high) != 3:
                continue
            values = [float(item) for item in low + high]
            if not all(math.isfinite(item) for item in values):
                continue
            if any(values[index + 3] < values[index] for index in range(3)):
                continue
            mins.append(values[:3])
            maxs.append(values[3:])
            node_ids.append(str(node.get("node_id", f"node_{len(node_ids)}")))
        if not mins:
            raise ValueError("scene has no usable node bounds")
        return cls(np.array(mins), np.array(maxs), node_ids)

    # -- queries ------------------------------------------------------------
    @property
    def bbox(self) -> tuple[list[float], list[float]]:
        return (
            [float(value) for value in self.mins.min(axis=0)],
            [float(value) for value in self.maxs.max(axis=0)],
        )

    def clearance(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Distance from each point to the nearest box surface (0 when inside)."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.empty(len(array), dtype=np.float64)
        for start in range(0, len(array), 16):
            chunk = array[start : start + 16][:, None, :]
            clamped = np.clip(chunk, self.mins[None, :, :], self.maxs[None, :, :])
            squared = ((chunk - clamped) ** 2).sum(axis=2)
            result[start : start + 16] = np.sqrt(squared.min(axis=1))
        return result

    def inside(self, point: Sequence[float], *, tolerance: float = 0.0) -> np.ndarray:
        values = np.asarray(point, dtype=np.float64)
        return np.all(
            (values >= self.mins - tolerance) & (values <= self.maxs + tolerance), axis=1
        )

    def nearby_mask(self, point: Sequence[float], radius: float) -> np.ndarray:
        """Mask of boxes reachable within ``radius`` of ``point`` (broad-phase cull)."""
        values = np.asarray(point, dtype=np.float64)[None, :]
        clamped = np.clip(values, self.mins, self.maxs)
        distance = np.sqrt(((values - clamped) ** 2).sum(axis=1))
        return distance <= radius

    def raycast(
        self,
        origin: Sequence[float],
        directions: np.ndarray,
        *,
        max_distance: float,
        mask: np.ndarray | None = None,
        chunk_rows: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cast rays against the box set.

        Returns ``(depth, box_index)`` where ``box_index`` is the global box index
        of the nearest hit, or ``-1`` where the ray hits nothing within
        ``max_distance``.
        """
        dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        mins, maxs, global_index = (
            (self.mins, self.maxs, np.arange(len(self.mins)))
            if mask is None
            else (self.mins[mask], self.maxs[mask], np.arange(len(self.mins))[mask])
        )
        depth = np.full(len(dirs), -1.0, dtype=np.float64)
        index = np.full(len(dirs), -1, dtype=np.int64)
        if not len(mins):
            return depth, index
        origin_row = np.asarray(origin, dtype=np.float64)[None, None, :]
        for start in range(0, len(dirs), chunk_rows):
            chunk = dirs[start : start + chunk_rows][:, None, :]
            parallel = np.abs(chunk) < 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                inverse = 1.0 / np.where(parallel, 1.0, chunk)
                near_plane = (mins[None, :, :] - origin_row) * inverse
                far_plane = (maxs[None, :, :] - origin_row) * inverse
            inside_slab = (mins[None, :, :] - origin_row <= 0.0) & (
                maxs[None, :, :] - origin_row >= 0.0
            )
            low = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), near_plane)
            high = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), far_plane)
            # Per axis the ray spans [min(low, high), max(low, high)]; the box is
            # hit when every axis interval overlaps: t_near = max of entry
            # distances, t_far = min of exit distances.
            t_near = np.minimum(low, high).max(axis=2)
            t_far = np.maximum(low, high).min(axis=2)
            entered = np.maximum(t_near, 0.0)
            hit = (t_far >= entered) & (t_near > 1e-6) & (t_near <= max_distance)
            candidate = np.where(hit, t_near, np.inf)
            local = np.argmin(candidate, axis=1)
            rows = np.arange(len(local))
            best = candidate[rows, local]
            found = np.isfinite(best)
            depth[start : start + len(local)] = np.where(found, best, -1.0)
            index[start : start + len(local)] = np.where(
                found, global_index[local], -1
            )
        return depth, index


# --------------------------------------------------------------------------- #
# Camera geometry
# --------------------------------------------------------------------------- #
def unit(value: Sequence[float], name: str = "vector") -> list[float]:
    length = math.sqrt(sum(item * item for item in value))
    if length < 1e-12:
        raise ValueError(f"{name} cannot be normalized")
    return [item / length for item in value]


def cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def camera_basis(yaw_degrees: float, pitch_degrees: float) -> tuple[list[float], list[float], list[float]]:
    """No-roll basis: ``forward`` from yaw/pitch, ``right`` horizontal, ``up`` completing it."""
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    forward = unit(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)],
        "camera forward",
    )
    right = unit(cross(forward, [0.0, 0.0, 1.0]), "camera right")
    up = unit(cross(right, forward), "camera up")
    return forward, right, up


def blender_rotation_euler(
    right: Sequence[float], up: Sequence[float], forward: Sequence[float]
) -> list[float]:
    """Blender XYZ euler for a no-roll camera looking along ``forward``.

    Matches ``coordinates.CameraFrame.verify_rotation``.  The world-from-camera
    rotation matrix has the columns ``right``, ``up`` and ``-forward``, so with
    ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``:

    ``R[2][0] = -sin(ry)``, ``R[2][1] = cos(ry) sin(rx)``,
    ``R[2][2] = cos(ry) cos(rx)``, ``R[0][0] = cos(rz) cos(ry)`` and
    ``R[1][0] = sin(rz) cos(ry)``.
    """
    ry = math.atan2(-right[2], math.hypot(right[0], right[1]))
    if math.hypot(right[0], right[1]) < 1e-9:
        # Looking straight up or down: only (rz - rx) and (rz + rx) are observable.
        return [0.0, ry, math.atan2(-up[0], up[1]) * (1.0 if math.cos(ry) > 0 else -1.0)]
    rx = math.atan2(up[2], -forward[2])
    rz = math.atan2(right[1], right[0])
    return [rx, ry, rz]


def view_directions(
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    spec: CameraSpec,
) -> np.ndarray:
    """Unit directions for a ``ray_grid``x``ray_grid`` sample of the frustum."""
    angle_x, angle_y = spec.angles()
    half_x = math.tan(angle_x * 0.5) * 1.02  # small guard band past the frame edge
    half_y = math.tan(angle_y * 0.5) * 1.02
    grid = max(3, int(spec.ray_grid))
    offsets = np.linspace(-1.0, 1.0, grid)
    forward_array = np.asarray(forward, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    directions = []
    for offset_y in offsets:
        for offset_x in offsets:
            direction = (
                forward_array + offset_x * half_x * right_array + offset_y * half_y * up_array
            )
            directions.append(direction / np.linalg.norm(direction))
    return np.asarray(directions)


# --------------------------------------------------------------------------- #
# Pose evaluation
# --------------------------------------------------------------------------- #
def evaluate_pose(
    free_space: FreeSpace,
    location: Sequence[float],
    yaw_degrees: float,
    pitch_degrees: float,
    spec: CameraSpec,
) -> dict[str, Any]:
    """Check one camera pose and return its metrics and accept/reject verdict.

    Rejection reasons are stable strings so callers can aggregate them.
    """
    point = [float(value) for value in location]
    forward, right, up = camera_basis(yaw_degrees, pitch_degrees)
    clearance = float(free_space.clearance([point])[0])
    metrics: dict[str, Any] = {
        "location": point,
        "yaw_degrees": float(yaw_degrees),
        "pitch_degrees": float(pitch_degrees),
        "forward": forward,
        "right": right,
        "up": up,
        "camera_clearance": clearance,
    }
    if clearance < spec.camera_radius:
        return {"ok": False, "reason": "camera_inside_geometry", "metrics": metrics}

    mask = free_space.nearby_mask(point, spec.max_depth)
    if not bool(mask.any()):
        return {"ok": False, "reason": "empty_view_volume", "metrics": metrics}

    directions = view_directions(forward, right, up, spec)
    depth, box_index = free_space.raycast(
        point, directions, max_distance=spec.max_depth, mask=mask
    )
    hits = depth >= 0.0
    hit_count = int(hits.sum())
    total = int(len(depth))
    hit_fraction = hit_count / max(total, 1)
    metrics["hit_fraction"] = hit_fraction
    metrics["ray_count"] = total
    if hit_fraction < spec.min_hit_fraction:
        return {"ok": False, "reason": "no_geometry_in_view", "metrics": metrics}

    hit_depths = depth[hits]
    minimum = float(hit_depths.min())
    metrics["minimum_hit_depth"] = minimum
    if minimum < spec.near_clearance:
        return {"ok": False, "reason": "near_geometry_in_view", "metrics": metrics}

    near_fraction = float(np.mean(hit_depths < spec.near_window))
    metrics["near_fraction"] = near_fraction
    if near_fraction > spec.max_near_fraction:
        return {"ok": False, "reason": "closeup_single_surface", "metrics": metrics}

    counts = np.bincount(box_index[hits], minlength=len(free_space.mins))
    dominant = int(np.argmax(counts))
    dominance = float(counts[dominant] / max(hit_count, 1))
    metrics["single_node_fraction"] = dominance
    metrics["dominant_node_id"] = free_space.node_ids[dominant]
    if dominance > spec.max_single_node_fraction:
        return {"ok": False, "reason": "single_node_dominates", "metrics": metrics}

    low = float(np.percentile(hit_depths, 10.0))
    high = float(np.percentile(hit_depths, 90.0))
    spread = high / max(low, 1e-6)
    metrics["depth_p10"] = low
    metrics["depth_p90"] = high
    metrics["depth_spread_ratio"] = spread
    if spread < spec.min_depth_spread_ratio:
        return {"ok": False, "reason": "depth_spread_too_small", "metrics": metrics}

    anchor_depth = float(np.median(hit_depths))
    anchor = [point[index] + forward[index] * anchor_depth for index in range(3)]
    metrics["anchor_depth"] = anchor_depth
    metrics["anchor"] = anchor
    metrics["score"] = (
        hit_fraction
        - 0.5 * dominance
        - 0.5 * near_fraction
        + 0.35 * min(math.log(max(spread, 1.0)), math.log(6.0)) / math.log(6.0)
        + 0.15 * min(clearance / max(spec.near_clearance, 1e-6), 2.0) / 2.0
    )
    return {"ok": True, "reason": None, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _seed_for(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _floor_boxes(scene: dict[str, Any], free_space: FreeSpace) -> np.ndarray:
    """Box indices whose entity is a stand-on surface (floor / sidewalk / stairs)."""
    node_index = {node_id: index for index, node_id in enumerate(free_space.node_ids)}
    selected: set[int] = set()
    for entity in scene.get("entities", []):
        if str(entity.get("category", "")) not in FLOOR_CATEGORIES:
            continue
        for node_id in entity.get("node_ids", []):
            index = node_index.get(str(node_id))
            if index is not None:
                selected.add(index)
    return np.array(sorted(selected), dtype=np.int64)


def sample_camera_poses(
    scene: dict[str, Any],
    spec: CameraSpec,
    *,
    seed: int,
    count: int,
    existing: Iterable[Sequence[float]] = (),
) -> dict[str, Any]:
    """Search ``count`` collision-free camera poses inside one scene.

    Returns ``{"accepted": [...], "rejects": [...], "stats": {...}}``.  Only the
    best-scoring accepted poses are returned when more than ``count`` qualify.
    """
    if count < 1:
        raise ValueError("count must be positive")
    free_space = FreeSpace.from_scene(scene)
    low, high = free_space.bbox
    rng = random.Random(_seed_for(seed, str(scene.get("scene_id", "scene")), "camera"))
    # Two-stage gate: a cheap ray grid rejects most poses, and only survivors are
    # re-verified at the full grid that ends up in the manifest.
    coarse_spec = replace(spec, ray_grid=max(3, min(spec.search_ray_grid, spec.ray_grid)))
    floors = _floor_boxes(scene, free_space)
    use_floor_policy = spec.height_policy == "floor_standing" and len(floors) > 0
    result_rejects: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    taken = [list(point) for point in existing]
    attempts = 0
    budget = int(count) * max(1, int(spec.attempts_per_camera)) * 4
    while len(accepted) < count and attempts < budget:
        attempts += 1
        x = rng.uniform(low[0], high[0])
        y = rng.uniform(low[1], high[1])
        if use_floor_policy:
            tops = free_space.maxs[floors, 2]
            selected = np.nonzero(
                (free_space.mins[floors, 0] <= x)
                & (free_space.maxs[floors, 0] >= x)
                & (free_space.mins[floors, 1] <= y)
                & (free_space.maxs[floors, 1] >= y)
            )[0]
            if not len(selected):
                result_rejects.append({"reason": "no_floor_under_candidate", "location": [x, y, None]})
                continue
            foot = float(tops[selected[rng.randrange(len(selected))]])
            z = foot + rng.uniform(*spec.eye_height)
        else:
            z = rng.uniform(
                low[2] + (high[2] - low[2]) * spec.free_height_band[0],
                low[2] + (high[2] - low[2]) * spec.free_height_band[1],
            )
        location = [x, y, z]
        if spec.min_camera_spacing > 0.0 and taken:
            nearest = min(
                math.dist(location, other) for other in taken
            )
            if nearest < spec.min_camera_spacing:
                result_rejects.append({"reason": "camera_too_close_to_other_camera", "location": location})
                continue
        yaw = rng.uniform(*spec.yaw_degrees)
        pitch = rng.uniform(*spec.pitch_degrees)
        verdict = evaluate_pose(free_space, location, yaw, pitch, coarse_spec)
        if not verdict["ok"]:
            result_rejects.append({"reason": verdict["reason"], "location": location})
            continue
        if coarse_spec.ray_grid != spec.ray_grid:
            verdict = evaluate_pose(free_space, location, yaw, pitch, spec)
            if not verdict["ok"]:
                result_rejects.append(
                    {"reason": f"final_gate:{verdict['reason']}", "location": location}
                )
                continue
        accepted.append(verdict["metrics"])
        taken.append(location)
    accepted.sort(key=lambda row: (-float(row["score"]), row["location"]))
    stats = {
        "attempts": attempts,
        "accepted": len(accepted),
        "kept": min(len(accepted), count),
        "acceptance_rate": (len(accepted) / attempts) if attempts else 0.0,
        "rejects": len(result_rejects),
        "reject_reasons": _count_reasons(result_rejects),
        "nodes": len(free_space.node_ids),
        "floor_nodes": int(len(floors)),
        "height_policy": "floor_standing" if use_floor_policy else "free",
    }
    return {"accepted": accepted[:count], "rejects": result_rejects, "stats": stats}


def _count_reasons(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# Job manifest
# --------------------------------------------------------------------------- #
def build_camera_job(
    scene: dict[str, Any],
    metrics: dict[str, Any],
    *,
    spec: CameraSpec,
    seed: int,
    index: int,
    base_scene_blend: str,
    base_scene_digest: str,
    lighting_profile: dict[str, Any],
    fixture_entity_ids: Sequence[str] = (),
    license_block: dict[str, Any] | None = None,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    location = [float(value) for value in metrics["location"]]
    forward = [float(value) for value in metrics["forward"]]
    right = [float(value) for value in metrics["right"]]
    up = [float(value) for value in metrics["up"]]
    target = [float(value) for value in metrics["anchor"]]
    angle_x, angle_y = spec.angles()
    width, height = (int(value) for value in spec.resolution)
    intrinsics = {
        "width": width,
        "height": height,
        "fx": (width * 0.5) / math.tan(angle_x * 0.5),
        "fy": (height * 0.5) / math.tan(angle_y * 0.5),
        "cx": width * 0.5,
        "cy": height * 0.5,
    }
    identity = {
        "contract_version": SCENE_CAMERA_CONTRACT_VERSION,
        "scene_id": scene.get("scene_id"),
        "base_scene_blend": base_scene_blend,
        "base_scene_digest": base_scene_digest,
        "seed": int(seed),
        "index": int(index),
        "camera": {
            "location": location,
            "target": target,
            "focal_length": float(spec.focal_length),
            "sensor_width": float(spec.sensor_width),
            "resolution": [width, height],
        },
        "lighting_profile": lighting_profile,
    }
    digest = _stable_digest(identity)
    job = {
        "schema_version": schema_version,
        "job_id": "scene_camera_" + digest.removeprefix("sha256:")[:20],
        "seed": _seed_for(seed, str(scene.get("scene_id")), str(index)),
        "scene_id": str(scene.get("scene_id")),
        "base_scene_blend": base_scene_blend,
        "base_scene_digest": base_scene_digest,
        "camera": {
            "strategy": "scene_free_space_v1",
            "location": location,
            "target": target,
            "rotation_euler": blender_rotation_euler(right, up, forward),
            "focal_length": float(spec.focal_length),
            "sensor_width": float(spec.sensor_width),
            "sensor_fit": "AUTO",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "angle_x": angle_x,
            "angle_y": angle_y,
            "resolution": {"width": width, "height": height},
            "intrinsics": intrinsics,
            "yaw_degrees": float(metrics["yaw_degrees"]),
            "pitch_degrees": float(metrics["pitch_degrees"]),
            "forward": forward,
            "up": up,
            "right": right,
            "camera_clearance": float(metrics["camera_clearance"]),
            "minimum_hit_depth": float(metrics["minimum_hit_depth"]),
            "hit_fraction": float(metrics["hit_fraction"]),
            "near_fraction": float(metrics["near_fraction"]),
            "depth_spread_ratio": float(metrics["depth_spread_ratio"]),
            "single_node_fraction": float(metrics["single_node_fraction"]),
            "score": float(metrics["score"]),
        },
        "canonical": {
            "origin": target,
            "asset_size": float(metrics["anchor_depth"]),
            "position_axes": CANONICAL_AXES,
            "units": "meter",
        },
        "camera_position_convention": {
            "coordinate_space": "opencv_camera",
            "origin": "camera.location",
            "axes": CAMERA_AXES,
            "depth": "signed_distance_along_camera_forward_axis",
        },
        "lighting_profile": dict(lighting_profile),
        "fixture_entity_ids": [str(value) for value in fixture_entity_ids],
        "license": dict(license_block or {}),
        "free_space_report": {
            "node_count": int(metrics.get("ray_count", 0)),
            "dominant_node_id": metrics.get("dominant_node_id"),
            "anchor_depth": float(metrics["anchor_depth"]),
            "depth_p10": float(metrics["depth_p10"]),
            "depth_p90": float(metrics["depth_p90"]),
        },
        "lineage": {
            "scene_digest": scene.get("source_digest"),
            "scene_camera_contract_version": SCENE_CAMERA_CONTRACT_VERSION,
            "camera_spec": spec.as_dict(),
            "scene_camera_digest": digest,
        },
    }
    return job


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def build_scene_camera_jobs(
    scene_document: dict[str, Any],
    *,
    spec: CameraSpec | None = None,
    seed: int = 0,
    cameras_per_scene: int = 1,
    lighting_profile: dict[str, Any] | None = None,
    blend_by_scene: dict[str, str] | None = None,
    fixture_categories: Iterable[str] = (),
    base_scene_license: dict[str, Any] | None = None,
    license_policy_version: str = "unconfigured",
    require_all_scenes: bool = True,
) -> dict[str, Any]:
    """Plan camera jobs for every scene in a ``scene.json`` document."""
    spec = spec or CameraSpec()
    profile = lighting_profile or {
        "name": "scene_native_v1",
        "ambient": "hdri_or_constant",
        "lights": "scene_native_fixtures_plus_frustum_points",
    }
    blend_by_scene = blend_by_scene or {}
    fixtures = {str(value) for value in fixture_categories}
    scene_license = dict(base_scene_license or {})
    license_block = {
        "base_scene": scene_license,
        "policy_version": str(license_policy_version),
        "decision": "allowed" if scene_license.get("decision") == "allowed" else "unverified",
    }
    scenes = sorted(scene_document.get("scenes", []), key=lambda row: row["scene_id"])
    jobs: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    per_scene: dict[str, Any] = {}
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        blend = blend_by_scene.get(scene_id, str(scene.get("normalized_blend", "")))
        digest = str(scene.get("source_digest", ""))
        fixture_ids = sorted(
            entity["entity_id"]
            for entity in scene.get("entities", [])
            if str(entity.get("category", "")) in fixtures
        )
        try:
            search = sample_camera_poses(scene, spec, seed=seed, count=cameras_per_scene)
        except ValueError as error:
            rejects.append({"scene_id": scene_id, "reason": "unusable_scene_bounds", "detail": str(error)})
            continue
        per_scene[scene_id] = search["stats"]
        if not search["accepted"]:
            rejects.append(
                {
                    "scene_id": scene_id,
                    "reason": "no_valid_camera_pose",
                    "reject_reasons": search["stats"]["reject_reasons"],
                }
            )
            continue
        for index, metrics in enumerate(search["accepted"]):
            jobs.append(
                build_camera_job(
                    scene,
                    metrics,
                    spec=spec,
                    seed=seed,
                    index=index,
                    base_scene_blend=blend,
                    base_scene_digest=digest,
                    lighting_profile=profile,
                    fixture_entity_ids=fixture_ids,
                    license_block=license_block,
                )
            )
    jobs.sort(key=lambda row: row["job_id"])
    document = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": "scene_camera_jobs_v1",
        "seed": int(seed),
        "cameras_per_scene": int(cameras_per_scene),
        "camera_spec": spec.as_dict(),
        "lighting_profile": profile,
        "jobs": jobs,
        "rejects": rejects,
        "stats": {
            "scenes": len(scenes),
            "scenes_without_camera": sum(1 for scene_id in per_scene if not per_scene[scene_id]["accepted"])
            + sum(1 for row in rejects if row["reason"] == "unusable_scene_bounds"),
            "jobs": len(jobs),
            "per_scene": per_scene,
        },
    }
    if require_all_scenes and not jobs:
        raise RuntimeError("scene camera planning produced zero jobs")
    return document


def load_scene_document(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "CameraSpec",
    "FreeSpace",
    "build_camera_job",
    "build_scene_camera_jobs",
    "blender_rotation_euler",
    "camera_basis",
    "evaluate_pose",
    "load_scene_document",
    "sample_camera_poses",
    "view_directions",
]
