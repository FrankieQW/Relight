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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .scene_targets import (
    SceneTarget,
    TargetSpec,
    look_angles,
    occluder_exclusions,
    project_points,
    radius_for,
    screen_bbox_ratio,
    select_targets,
)


SCENE_CAMERA_CONTRACT_VERSION = "1"
CANONICAL_AXES = "x=right,y=camera-forward,z=up"  # must match coordinates.CANONICAL_AXES
CAMERA_AXES = "x=right,y=down,z=forward"

# Entities that can be stood on.  ``scene_extract`` already labels these, so the
# eye-height policy costs nothing extra.
FLOOR_CATEGORIES = frozenset(
    {"floor", "ground", "sidewalk", "road", "stair", "stairs", "staircase", "landing", "platform"}
)


@dataclass(frozen=True)
class ContainerSpec:
    """How a *hollow* node (a room / building shell) is recognised and used.

    The AABB free-space model has one fatal assumption indoors: a box is solid.
    An imported interior often ships as a single node - ``paris_building_01_
    interior``, 877 m3 - whose AABB covers the whole room, so every eye-height
    point is "inside geometry" and the room itself never produces a ray hit.
    Nothing can be planned until that node is treated as a hollow shell.

    ``model = "aabb_legacy"`` restores the pre-container behaviour exactly, which
    is what reproducing an old dataset needs.
    """

    model: str = "container_v1"  # or "aabb_legacy"
    min_extent: float = 2.0  # metres; all three axes (a table is 0.75 m tall)
    min_volume: float = 8.0  # cubic metres
    min_contained_nodes: int = 1  # other node centres inside the AABB
    clearance_margin: float = 0.10  # extra wall distance for the AABB approximation

    @property
    def active(self) -> bool:
        return str(self.model) != "aabb_legacy"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": str(self.model),
            "min_extent": float(self.min_extent),
            "min_volume": float(self.min_volume),
            "min_contained_nodes": int(self.min_contained_nodes),
            "clearance_margin": float(self.clearance_margin),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ContainerSpec":
        source = dict(value or {})
        defaults = cls()
        model = str(source.pop("model", defaults.model))
        if model not in {"container_v1", "aabb_legacy"}:
            raise ValueError(
                f"m4.camera.container.model is not implemented: {model!r} "
                "(expected container_v1 or aabb_legacy)"
            )
        return cls(
            model=model,
            min_extent=float(source.get("min_extent", defaults.min_extent)),
            min_volume=float(source.get("min_volume", defaults.min_volume)),
            min_contained_nodes=int(
                source.get("min_contained_nodes", defaults.min_contained_nodes)
            ),
            clearance_margin=float(
                source.get("clearance_margin", defaults.clearance_margin)
            ),
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
    # Absolute clamp (metres above the scene bbox floor) applied on top of
    # ``free_height_band``.  Without it, a whole-building bbox puts cameras up to
    # 8.9 m in the air, i.e. above the roof, looking at sky.
    free_height_clamp: tuple[float, float] = (1.00, 2.50)
    container: ContainerSpec = ContainerSpec()

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

    # Strategy.  ``scene_free_space_v1`` is the original planner and stays
    # bit-for-bit reproducible; ``entity_conditioned_v1`` frames a named target
    # entity and mixes in ``mix_with_free_space`` room-level wide shots.
    strategy: str = "scene_free_space_v1"
    mix_with_free_space: float = 0.30
    min_hit_fraction_targeted: float = 0.60
    # ``near_window``/``max_near_fraction`` are room-scale: at a 0.6 m orbit the
    # table and floor are *always* inside 1 m, so close-ups need their own cap.
    max_near_fraction_targeted: float = 0.85
    target: TargetSpec = TargetSpec()

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
            "free_height_clamp": [float(value) for value in self.free_height_clamp],
            "container": self.container.as_dict(),
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
            "strategy": self.strategy,
            "mix_with_free_space": float(self.mix_with_free_space),
            "min_hit_fraction_targeted": float(self.min_hit_fraction_targeted),
            "max_near_fraction_targeted": float(self.max_near_fraction_targeted),
            "target": self.target.as_dict(),
            "angle_x": angle_x,
            "angle_y": angle_y,
        }

    def tan_half_x(self) -> float:
        """``tan(angle_x / 2)``; used to estimate a target's on-screen size."""
        angle_x, _angle_y = self.angles()
        return math.tan(angle_x * 0.5)


# --------------------------------------------------------------------------- #
# Free space
# --------------------------------------------------------------------------- #
def detect_containers(
    mins: np.ndarray, maxs: np.ndarray, spec: ContainerSpec | None = None
) -> np.ndarray:
    """Flag the boxes that are room/building *shells* rather than solid objects.

    Three conditions, all cheap:

    * at least ``min_extent`` in every axis - a table is 0.75 m tall, a wardrobe
      1.0 m deep, so neither can qualify; only a room-like volume can;
    * at least ``min_volume`` - rules out a tall thin pillar or a doorway gap;
    * the AABB contains at least ``min_contained_nodes`` other node centres -
      a solid 2x2x2 crate contains nothing, a furnished room contains the
      furniture.

    Category is deliberately not consulted: ``paris_building_01_interior`` only
    classifies as ``building`` because ``building`` happens to be a generic
    token, and relying on that would break on the next dataset.
    """
    spec = spec or ContainerSpec()
    flags = np.zeros(len(mins), dtype=bool)
    if not spec.active or not len(mins):
        return flags
    dims = np.maximum(maxs - mins, 0.0)
    volume = np.prod(dims, axis=1)
    plausible = (dims.min(axis=1) >= float(spec.min_extent)) & (
        volume >= float(spec.min_volume)
    )
    candidates = np.nonzero(plausible)[0]
    if not len(candidates):
        return flags
    centers = (mins + maxs) * 0.5
    needed = max(1, int(spec.min_contained_nodes))
    for index in candidates:
        inside = np.all(
            (centers >= mins[index]) & (centers <= maxs[index]), axis=1
        )
        inside[index] = False
        if int(inside.sum()) >= needed:
            flags[index] = True
    return flags


class FreeSpace:
    """Axis-aligned occlusion volume built from ``scene.json`` node bounds.

    Boxes are solid by default.  A box big enough to be a room shell is marked
    *hollow* (see :class:`ContainerSpec`), which changes three things: a point
    inside it is measured to its nearest face instead of being "inside geometry",
    a ray leaving it hits the far face instead of passing through, and its hits
    do not count as "one object dominating the frame".
    """

    def __init__(
        self,
        mins: np.ndarray,
        maxs: np.ndarray,
        node_ids: Sequence[str],
        containers: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
        if mins.shape != maxs.shape or mins.ndim != 2 or mins.shape[1] != 3:
            raise ValueError("free-space bounds must be (N, 3) arrays")
        if len(node_ids) != len(mins):
            raise ValueError("one node id is required per bounding box")
        self.mins = np.asarray(mins, dtype=np.float64)
        self.maxs = np.asarray(maxs, dtype=np.float64)
        self.node_ids = list(node_ids)
        if containers is None:
            flag = np.zeros(len(self.mins), dtype=bool)
        else:
            flag = np.asarray(containers, dtype=bool)
            if flag.shape != (len(self.mins),):
                raise ValueError("container flags must have one entry per box")
        self.containers = flag
        self.volume = np.prod(np.maximum(self.maxs - self.mins, 0.0), axis=1)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_scene(
        cls,
        scene: dict[str, Any],
        container: ContainerSpec | None = None,
    ) -> "FreeSpace":
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
        spec = container or ContainerSpec()
        flags = detect_containers(
            np.array(mins, dtype=np.float64), np.array(maxs, dtype=np.float64), spec
        )
        return cls(np.array(mins), np.array(maxs), node_ids, flags)

    # -- queries ------------------------------------------------------------
    @property
    def bbox(self) -> tuple[list[float], list[float]]:
        return (
            [float(value) for value in self.mins.min(axis=0)],
            [float(value) for value in self.maxs.max(axis=0)],
        )

    @property
    def container_node_ids(self) -> list[str]:
        return [self.node_ids[index] for index in np.nonzero(self.containers)[0]]

    def container_at(self, point: Sequence[float]) -> int:
        """Index of the smallest hollow box containing ``point`` (``-1`` if none)."""
        if not bool(self.containers.any()):
            return -1
        values = np.asarray(point, dtype=np.float64)
        inside = np.all((values >= self.mins) & (values <= self.maxs), axis=1)
        inside &= self.containers
        candidates = np.nonzero(inside)[0]
        if not len(candidates):
            return -1
        return int(candidates[np.argmin(self.volume[candidates])])

    def primary_container(self) -> int:
        """Largest hollow box: the room the fragment is about (``-1`` if none)."""
        if not bool(self.containers.any()):
            return -1
        candidates = np.nonzero(self.containers)[0]
        return int(candidates[np.argmax(self.volume[candidates])])

    def outside_room(self, point: Sequence[float]) -> bool:
        """True when ``point`` is above the roof / below the floor of a room.

        Horizontally inside a container but vertically outside its z band is
        unambiguously wrong (that is how a camera ends up looking at sky from
        8.9 m up), while a point that is horizontally *outside* every container
        is left alone so exterior parts of a mixed scene still work.
        """
        if not bool(self.containers.any()):
            return False
        values = np.asarray(point, dtype=np.float64)
        footprint = (
            (values[0] >= self.mins[:, 0])
            & (values[0] <= self.maxs[:, 0])
            & (values[1] >= self.mins[:, 1])
            & (values[1] <= self.maxs[:, 1])
            & self.containers
        )
        in_band = (values[2] >= self.mins[:, 2]) & (values[2] <= self.maxs[:, 2])
        return bool(np.any(footprint & ~in_band))

    def clearance(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Distance from each point to the nearest *solid* surface.

        Outside every box this is the euclidean gap; inside a solid box it is 0
        (the point is in geometry); inside a hollow box it is the distance to the
        nearest face, i.e. the distance to the wall, which is what a camera in a
        room needs to know.
        """
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.empty(len(array), dtype=np.float64)
        has_containers = bool(self.containers.any())
        for start in range(0, len(array), 16):
            chunk = array[start : start + 16][:, None, :]
            gap = np.maximum(self.mins[None, :, :] - chunk, chunk - self.maxs[None, :, :])
            inside = np.all(gap <= 0.0, axis=2)
            outside = np.sqrt((np.maximum(gap, 0.0) ** 2).sum(axis=2))
            if has_containers:
                to_face = np.minimum(
                    chunk - self.mins[None, :, :], self.maxs[None, :, :] - chunk
                ).min(axis=2)
                solid_inside = inside & ~self.containers[None, :]
                hollow_inside = inside & self.containers[None, :]
                value = np.where(
                    hollow_inside, to_face, np.where(solid_inside, 0.0, outside)
                )
            else:
                value = np.where(inside, 0.0, outside)
            result[start : start + 16] = value.min(axis=1)
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
        mins, maxs, global_index, containers = (
            (self.mins, self.maxs, np.arange(len(self.mins)), self.containers)
            if mask is None
            else (
                self.mins[mask],
                self.maxs[mask],
                np.arange(len(self.mins))[mask],
                self.containers[mask],
            )
        )
        depth = np.full(len(dirs), -1.0, dtype=np.float64)
        index = np.full(len(dirs), -1, dtype=np.int64)
        if not len(mins):
            return depth, index
        origin_row = np.asarray(origin, dtype=np.float64)[None, None, :]
        hollow = bool(containers.any())
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
            # A ray parallel to an axis and outside that slab can never enter the
            # box, but min(inf, -inf) = -inf turns the empty interval into a full
            # one and the box is reported hit.  Only exactly axis-parallel rays
            # (pitch 0 rows of the grid) trigger this, which is why it survived.
            empty_axis = parallel & ~inside_slab
            t_near = np.where(empty_axis.any(axis=2), np.inf, t_near)
            entered = np.maximum(t_near, 0.0)
            hit = (t_far >= entered) & (t_near > 1e-6) & (t_near <= max_distance)
            if hollow:
                # A hollow box the ray starts *inside* is entered at t_near <= 0;
                # the visible surface is the far face, i.e. the opposite wall.
                # Without this the room contributes no geometry at all from
                # inside, which is why interior views scored hit_fraction 0.
                exit_hit = (
                    (t_near <= 1e-6)
                    & (t_far > 1e-6)
                    & (t_far <= max_distance)
                    & containers[None, :]
                )
                hit = hit | exit_hit
                entered = np.where(exit_hit, t_far, t_near)
            candidate = np.where(hit, entered, np.inf)
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
    container = free_space.container_at(point)
    metrics: dict[str, Any] = {
        "location": point,
        "yaw_degrees": float(yaw_degrees),
        "pitch_degrees": float(pitch_degrees),
        "forward": forward,
        "right": right,
        "up": up,
        "camera_clearance": clearance,
        "room_node_id": free_space.node_ids[container] if container >= 0 else None,
        "container_node_ids": free_space.container_node_ids,
    }
    if free_space.outside_room(point):
        return {"ok": False, "reason": "camera_outside_room", "metrics": metrics}
    # Inside a room the clearance comes from a rectangle approximation of the
    # shell, so keep an extra margin: overestimating the wall distance is the
    # failure mode that makes the renderer reject a pose the planner accepted.
    required = float(spec.camera_radius) + (
        float(spec.container.clearance_margin) if container >= 0 else 0.0
    )
    if clearance < required:
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
    if container >= 0:
        # The room is the environment, not "one object filling the frame": all
        # rays in an interior view end on the shell, so counting it would make
        # every good interior pose fail single_node_dominates.
        counts[container] = 0
    dominant = int(np.argmax(counts))
    dominance = (
        float(counts[dominant] / max(hit_count, 1)) if counts[dominant] > 0 else 0.0
    )
    metrics["single_node_fraction"] = dominance
    metrics["dominant_node_id"] = (
        free_space.node_ids[dominant] if counts[dominant] > 0 else None
    )
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


def target_screen_metrics(
    location: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
    spec: CameraSpec,
    target: SceneTarget,
    *,
    margin: float | None = None,
) -> dict[str, Any]:
    """On-screen size and centring of one target from a given pose."""
    angle_x, angle_y = spec.angles()
    limit = float(spec.target.in_frame_margin if margin is None else margin)
    u, v, depth = project_points(
        location, forward, right, up, angle_x, angle_y, target.samples
    )
    front = depth > 1e-6
    if not bool(front.any()):
        return {"screen_ratio": 0.0, "in_frame": False, "front_samples": 0}
    center_u, center_v, center_depth = project_points(
        location, forward, right, up, angle_x, angle_y, [target.center]
    )
    in_frame = bool(
        float(center_depth[0]) > 1e-6
        and abs(float(center_u[0])) <= limit
        and abs(float(center_v[0])) <= limit
    )
    return {
        "screen_ratio": screen_bbox_ratio(u[front], v[front]),
        "in_frame": in_frame,
        "front_samples": int(front.sum()),
    }


def evaluate_target_pose(
    free_space: FreeSpace,
    target: SceneTarget,
    location: Sequence[float],
    yaw_degrees: float,
    pitch_degrees: float,
    spec: CameraSpec,
    *,
    non_occluders: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Evaluate a pose that is supposed to show ``target``.

    The documented rejection reasons are stable strings.  Compared with
    :func:`evaluate_pose` the emphasis flips: the target is *allowed* to dominate
    the frame, and the hard statistics (near plane, dominance, depth spread) are
    computed over non-target geometry only.  What must hold instead is that the
    target is genuinely visible and reasonably sized.
    """
    point = [float(value) for value in location]
    forward, right, up = camera_basis(yaw_degrees, pitch_degrees)
    clearance = float(free_space.clearance([point])[0])
    container = free_space.container_at(point)
    metrics: dict[str, Any] = {
        "location": point,
        "yaw_degrees": float(yaw_degrees),
        "pitch_degrees": float(pitch_degrees),
        "forward": forward,
        "right": right,
        "up": up,
        "camera_clearance": clearance,
        "target_entity_id": target.entity_id,
        "target_label": target.label,
        "target_category": target.category,
        "target_node_ids": list(target.node_ids),
        "target_center": [float(value) for value in target.center],
        "target_radius": float(target.half_diagonal),
        "room_node_id": free_space.node_ids[container] if container >= 0 else None,
        "container_node_ids": free_space.container_node_ids,
    }
    if free_space.outside_room(point):
        return {"ok": False, "reason": "camera_outside_room", "metrics": metrics}
    required = float(spec.camera_radius) + (
        float(spec.container.clearance_margin) if container >= 0 else 0.0
    )
    if clearance < required:
        return {"ok": False, "reason": "camera_inside_geometry", "metrics": metrics}

    node_index = {node_id: index for index, node_id in enumerate(free_space.node_ids)}
    target_indices = [
        node_index[node_id] for node_id in target.node_ids if node_id in node_index
    ]
    non_occluder_indices = [
        node_index[node_id] for node_id in non_occluders if node_id in node_index
    ]
    mask = free_space.nearby_mask(point, spec.max_depth)
    if not bool(mask.any()):
        return {"ok": False, "reason": "empty_view_volume", "metrics": metrics}
    if not target_indices:
        return {"ok": False, "reason": "target_not_in_scene", "metrics": metrics}

    # -- target visibility --------------------------------------------------
    samples = np.asarray(target.samples, dtype=np.float64).reshape(-1, 3)
    if not len(samples):
        return {"ok": False, "reason": "target_has_no_samples", "metrics": metrics}
    relative = samples - np.asarray(point, dtype=np.float64)[None, :]
    distances = np.linalg.norm(relative, axis=1)
    usable = distances > 1e-6
    if not bool(usable.any()):
        return {"ok": False, "reason": "target_has_no_samples", "metrics": metrics}
    directions = relative[usable] / distances[usable][:, None]
    sample_depths, sample_boxes = free_space.raycast(
        point, directions, max_distance=spec.max_depth, mask=mask
    )
    own = np.isin(sample_boxes, np.asarray(target_indices, dtype=np.int64))
    hits = sample_depths >= 0.0
    # A sample whose ray is stopped by a non-occluding thin/transparent node, or
    # by a surface it physically rests on, still counts as visible: the cup on
    # the table is not "hidden by the table", it is touching it.
    tolerated = hits & (np.abs(distances[usable] - sample_depths) <= float(spec.target.contact_tolerance))
    transparent = hits & np.isin(
        sample_boxes, np.asarray(non_occluder_indices, dtype=np.int64)
    )
    visible = own | tolerated | transparent
    visible_ratio = float(visible.sum() / max(len(visible), 1))
    metrics["target_visible_ratio"] = visible_ratio
    metrics["target_sample_count"] = int(len(visible))

    occluders = sample_boxes[hits & ~own & ~tolerated & ~transparent]
    metrics["occluder_node_ids"] = sorted(
        {
            free_space.node_ids[int(index)]
            for index in np.unique(occluders)
            if int(index) >= 0
        }
    )
    if visible_ratio < float(spec.target.min_target_visible_ratio):
        return {"ok": False, "reason": "target_occluded", "metrics": metrics}

    # -- framing ------------------------------------------------------------
    framing = target_screen_metrics(point, forward, right, up, spec, target)
    metrics["target_screen_ratio"] = float(framing["screen_ratio"])
    metrics["target_in_frame"] = bool(framing["in_frame"])
    if not framing["in_frame"]:
        return {"ok": False, "reason": "target_out_of_frame", "metrics": metrics}
    if framing["screen_ratio"] < float(spec.target.min_target_screen_ratio):
        return {"ok": False, "reason": "target_too_small", "metrics": metrics}
    if framing["screen_ratio"] > float(spec.target.max_target_screen_ratio):
        return {"ok": False, "reason": "target_too_large", "metrics": metrics}

    # -- legacy gates, computed over non-target geometry --------------------
    directions = view_directions(forward, right, up, spec)
    depth, box_index = free_space.raycast(
        point, directions, max_distance=spec.max_depth, mask=mask
    )
    hits = depth >= 0.0
    hit_count = int(hits.sum())
    total = int(len(depth))
    metrics["ray_count"] = total
    metrics["hit_fraction"] = float(hit_count / max(total, 1))
    if metrics["hit_fraction"] < float(spec.min_hit_fraction_targeted):
        return {"ok": False, "reason": "no_geometry_in_view", "metrics": metrics}

    target_hit = hits & np.isin(box_index, np.asarray(target_indices, dtype=np.int64))
    others = hits & ~target_hit
    other_depths = depth[others]
    if other_depths.size >= 4:
        minimum = float(other_depths.min())
        metrics["minimum_hit_depth"] = minimum
        if minimum < spec.near_clearance:
            return {"ok": False, "reason": "near_geometry_in_view", "metrics": metrics}
        near_fraction = float(np.mean(other_depths < spec.near_window))
        counts = np.bincount(box_index[others], minlength=len(free_space.mins))
        if container >= 0:
            # The shell is the room; counting it as "the dominant object" would
            # reject every legitimate interior framing.
            counts[container] = 0
        dominant = int(np.argmax(counts))
        dominance = (
            float(counts[dominant] / max(int(others.sum()), 1))
            if counts[dominant] > 0
            else 0.0
        )
        low = float(np.percentile(other_depths, 10.0))
        high = float(np.percentile(other_depths, 90.0))
        spread = high / max(low, 1e-6)
        metrics["dominant_node_id"] = (
            free_space.node_ids[dominant] if counts[dominant] > 0 else None
        )
    else:
        # The target legitimately fills the frame; nothing else is visible.
        minimum = float(depth[hits].min()) if hit_count else float(spec.max_depth)
        metrics["minimum_hit_depth"] = minimum
        near_fraction = 0.0
        dominance = 0.0
        low = high = minimum
        spread = float("inf")
        metrics["dominant_node_id"] = target.node_ids[0] if target.node_ids else None
    metrics["near_fraction"] = near_fraction
    metrics["single_node_fraction"] = dominance
    metrics["depth_p10"] = low
    metrics["depth_p90"] = high
    metrics["depth_spread_ratio"] = spread
    if near_fraction > spec.max_near_fraction_targeted:
        return {"ok": False, "reason": "closeup_single_surface", "metrics": metrics}
    if dominance > spec.max_single_node_fraction:
        return {"ok": False, "reason": "single_node_dominates", "metrics": metrics}
    if math.isfinite(spread) and spread < spec.min_depth_spread_ratio:
        return {"ok": False, "reason": "depth_spread_too_small", "metrics": metrics}

    # -- anchor: the target centre, so canonical.origin is a real place -----
    anchor_depth = float(
        math.dist(point, [float(value) for value in target.center])
    )
    metrics["anchor_depth"] = anchor_depth
    metrics["anchor"] = [float(value) for value in target.center]
    metrics["score"] = (
        0.45 * visible_ratio
        + 0.25 * min(framing["screen_ratio"] / 0.15, 1.0)
        - 0.50 * near_fraction
        - 0.30 * dominance
        + 0.10 * metrics["hit_fraction"]
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
    targets: Sequence[SceneTarget] = (),
    min_target_screen_ratio: float | None = None,
) -> dict[str, Any]:
    """Search ``count`` collision-free camera poses inside one scene.

    Returns ``{"accepted": [...], "rejects": [...], "stats": {...}}``.  Only the
    best-scoring accepted poses are returned when more than ``count`` qualify.
    When ``targets`` and ``min_target_screen_ratio`` are supplied, a pose is only
    kept if at least one target reaches that on-screen size - so even the
    room-level wide shots cannot degenerate into a bare wall corner.
    """
    if count < 1:
        raise ValueError("count must be positive")
    free_space = FreeSpace.from_scene(scene, spec.container)
    low, high = free_space.bbox
    rng = random.Random(_seed_for(seed, str(scene.get("scene_id", "scene")), "camera"))
    # Two-stage gate: a cheap ray grid rejects most poses, and only survivors are
    # re-verified at the full grid that ends up in the manifest.
    coarse_spec = replace(spec, ray_grid=max(3, min(spec.search_ray_grid, spec.ray_grid)))
    floors = _floor_boxes(scene, free_space)
    # Where "the floor" comes from, in priority order:
    #   1. an entity the extractor already labelled floor/ground/road/...;
    #   2. the bottom face of the room shell - the floor of an imported interior
    #      is part of the shell mesh, so there is no floor *entity* indoors and
    #      the old code silently fell through to the free band;
    #   3. a clamped free band (outdoor / floorless fragments).
    room = free_space.primary_container()
    use_floor_policy = spec.height_policy == "floor_standing" and len(floors) > 0
    use_room_floor = (
        spec.height_policy == "floor_standing" and not use_floor_policy and room >= 0
    )
    if use_floor_policy:
        floor_source, floor_z = "support_categories", None
    elif use_room_floor:
        floor_source, floor_z = "room_shell", float(free_space.mins[room, 2])
    else:
        floor_source, floor_z = "free_band", None
    result_rejects: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    taken = [list(point) for point in existing]
    attempts = 0
    budget = int(count) * max(1, int(spec.attempts_per_camera)) * 4
    while len(accepted) < count and attempts < budget:
        attempts += 1
        if use_room_floor:
            # Sample inside the room footprint: the fragment bbox may be larger
            # than the room, and a point outside the walls is not a camera pose.
            x = rng.uniform(float(free_space.mins[room, 0]), float(free_space.maxs[room, 0]))
            y = rng.uniform(float(free_space.mins[room, 1]), float(free_space.maxs[room, 1]))
        else:
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
        elif use_room_floor:
            z = float(floor_z) + rng.uniform(*spec.eye_height)
        else:
            band_low = low[2] + (high[2] - low[2]) * spec.free_height_band[0]
            band_high = low[2] + (high[2] - low[2]) * spec.free_height_band[1]
            if spec.container.active:
                # A whole-building bbox reaches 20 m; the free band fraction then
                # puts cameras above the roof, staring at sky.
                clamp_low = low[2] + float(spec.free_height_clamp[0])
                clamp_high = low[2] + float(spec.free_height_clamp[1])
                band_low = max(band_low, clamp_low)
                band_high = min(band_high, clamp_high)
                if band_high <= band_low:
                    band_low = band_high = clamp_low
            z = rng.uniform(band_low, band_high)
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
        if targets and min_target_screen_ratio is not None:
            metrics = verdict["metrics"]
            best = max(
                target_screen_metrics(
                    location,
                    metrics["forward"],
                    metrics["right"],
                    metrics["up"],
                    spec,
                    target,
                )["screen_ratio"]
                for target in targets
            )
            if best < float(min_target_screen_ratio):
                result_rejects.append(
                    {"reason": "no_target_in_view", "location": location}
                )
                continue
        accepted.append(verdict["metrics"])
        taken.append(location)
    accepted.sort(key=lambda row: (-float(row["score"]), row["location"]))
    z_values = sorted(float(row["location"][2]) for row in accepted)
    stats = {
        "attempts": attempts,
        "accepted": len(accepted),
        "kept": min(len(accepted), count),
        "acceptance_rate": (len(accepted) / attempts) if attempts else 0.0,
        "rejects": len(result_rejects),
        "reject_reasons": _count_reasons(result_rejects),
        "nodes": len(free_space.node_ids),
        "floor_nodes": int(len(floors)),
        "floor_source": floor_source,
        "floor_z": None if floor_z is None else float(floor_z),
        "height_policy": "floor_standing" if (use_floor_policy or use_room_floor) else "free",
        "rooms": free_space.container_node_ids,
        "camera_z_percentiles": _percentiles(z_values),
    }
    return {"accepted": accepted[:count], "rejects": result_rejects, "stats": stats}


def _percentiles(values: Sequence[float]) -> list[float] | None:
    """``[p10, p50, p90]`` of a sorted list, or ``None`` when it is empty."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    picks = []
    for fraction in (0.10, 0.50, 0.90):
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        picks.append(round(ordered[index], 3))
    return picks


def _local_radius_cap(
    free_space: FreeSpace,
    center: Sequence[float],
    target_node_ids: Sequence[str],
    spec: CameraSpec,
    *,
    base_radius: float | None = None,
) -> float:
    """Upper bound on the orbit radius so the shell does not sink into geometry.

    An object pushed against a wall or standing on a table has almost no free
    space on some sides; clamping the radius keeps the sampler from spending its
    whole budget on candidates that are already inside a mesh.

    Two details matter indoors: the room shell must not count as an obstacle
    (it contains the target, so the distance would be 0 and the cap would
    collapse to the minimum, jamming every camera against a 1.2 m table), and
    the cap never drops below a fraction of the natural radius.
    """
    node_index = {node_id: index for index, node_id in enumerate(free_space.node_ids)}
    own = [node_index[value] for value in target_node_ids if value in node_index]
    others = np.ones(len(free_space.mins), dtype=bool)
    if own:
        others[own] = False
    others[free_space.containers] = False
    base = (
        float(base_radius)
        if base_radius is not None
        else float(spec.target.min_radius)
    )
    if not bool(others.any()):
        return float(spec.target.max_radius)
    point = np.asarray(center, dtype=np.float64)[None, :]
    clamped = np.clip(point, free_space.mins[others], free_space.maxs[others])
    distance = float(np.sqrt(((point - clamped) ** 2).sum(axis=1)).min())
    upper = max(
        float(spec.target.min_radius),
        float(spec.target.min_radius_fraction) * base,
        0.8 * distance,
    )
    return float(min(float(spec.target.max_radius), upper))


def sample_targeted_poses(
    scene: dict[str, Any],
    spec: CameraSpec,
    *,
    seed: int,
    count: int,
    existing: Iterable[Sequence[float]] = (),
) -> dict[str, Any]:
    """Orbit ``count`` cameras around the scene's best framable targets.

    Structure mirrors :func:`sample_camera_poses` so callers can treat both the
    same.  The extra ``targets`` key carries the chosen :class:`SceneTarget`
    objects for reuse (e.g. the room-level wide shots must still contain one).
    """
    if count < 1:
        raise ValueError("count must be positive")
    stats_empty = {
        "attempts": 0,
        "accepted": 0,
        "kept": 0,
        "acceptance_rate": 0.0,
        "rejects": 0,
        "reject_reasons": {},
        "nodes": len(scene.get("nodes") or []),
        "floor_nodes": 0,
        "height_policy": "entity_orbit",
        "targets": 0,
        "target_ids": [],
    }
    if not spec.target.enabled:
        return {
            "accepted": [],
            "rejects": [{"reason": "targets_disabled"}],
            "stats": stats_empty,
            "targets": [],
        }
    try:
        free_space = FreeSpace.from_scene(scene, spec.container)
    except ValueError as error:
        return {
            "accepted": [],
            "rejects": [{"reason": "unusable_scene_bounds", "detail": str(error)}],
            "stats": stats_empty,
            "targets": [],
        }
    room = free_space.primary_container()
    floor_z = float(free_space.mins[room, 2]) if room >= 0 else None
    target_report: dict[str, Any] = {}
    targets = select_targets(
        scene,
        spec.target,
        seed=seed,
        tan_half_x=spec.tan_half_x(),
        floor_z=floor_z,
        report=target_report,
    )
    if not targets:
        stats_empty = dict(stats_empty)
        stats_empty["targets_report"] = target_report
        stats_empty["rooms"] = free_space.container_node_ids
        return {
            "accepted": [],
            "rejects": [{"reason": "no_valid_target"}],
            "stats": stats_empty,
            "targets": [],
        }
    non_occluders = occluder_exclusions(scene, spec.target)
    rng = random.Random(_seed_for(seed, str(scene.get("scene_id", "scene")), "target"))
    accepted: list[dict[str, Any]] = []
    result_rejects: list[dict[str, Any]] = []
    taken = [list(point) for point in existing]
    # Azimuths already used, per target.  Orbiting the *same* subject, spacing
    # has to be angular: two cameras on a 0.6 m orbit are at most 1.2 m apart, so
    # a 1.5 m euclidean spacing allows exactly one camera per target.
    used_azimuths: dict[str, list[float]] = {}
    separation = float(spec.target.min_azimuth_separation_degrees)
    radius_low, radius_high = spec.target.radius_jitter
    radius_low = max(float(radius_low), 1e-3)
    radius_high = max(float(radius_high), radius_low)
    azimuth_count = max(1, int(spec.target.azimuth_samples))
    budget = int(count) * max(1, int(spec.target.attempts_per_target))
    attempts = 0
    cursor = 0
    while len(accepted) < count and attempts < budget:
        target = targets[cursor % len(targets)]
        cursor += 1
        attempts += 1
        azimuth = (
            360.0 * float(attempts % azimuth_count) / float(azimuth_count)
            + rng.uniform(
                -float(spec.target.azimuth_jitter_degrees),
                float(spec.target.azimuth_jitter_degrees),
            )
        )
        pitch = rng.uniform(*spec.target.pitch_degrees)
        yawrad = math.radians(azimuth)
        pitchrad = math.radians(pitch)
        direction = (
            math.cos(pitchrad) * math.cos(yawrad),
            math.cos(pitchrad) * math.sin(yawrad),
            math.sin(pitchrad),
        )
        base_radius = radius_for(target.half_diagonal, spec.target)
        cap = _local_radius_cap(
            free_space, target.center, target.node_ids, spec, base_radius=base_radius
        )
        base = base_radius
        scale = math.exp(
            rng.uniform(math.log(radius_low), math.log(radius_high))
        )
        radius = max(
            float(spec.target.min_radius),
            min(cap, base * scale),
        )
        location = [
            float(target.center[axis]) + direction[axis] * radius for axis in range(3)
        ]
        # Only the caller-provided poses need a euclidean check; poses accepted
        # here are all on the same orbit and are separated by azimuth instead.
        if spec.min_camera_spacing > 0.0 and taken:
            nearest = min(math.dist(location, other) for other in taken)
            if nearest < spec.min_camera_spacing:
                result_rejects.append(
                    {"reason": "camera_too_close_to_other_camera", "location": location}
                )
                continue
        if separation > 0.0:
            previous = used_azimuths.get(target.entity_id, [])
            if any(
                abs(((azimuth - other + 180.0) % 360.0) - 180.0) < separation
                for other in previous
            ):
                result_rejects.append(
                    {"reason": "camera_azimuth_too_close", "location": location}
                )
                continue
        to_center = [-(location[axis] - float(target.center[axis])) for axis in range(3)]
        yaw, pitch_degrees = look_angles(to_center)
        if spec.target.look_jitter_degrees > 0.0:
            jitter = float(spec.target.look_jitter_degrees)
            yaw += rng.uniform(-jitter, jitter)
            pitch_degrees += rng.uniform(-jitter, jitter)
        verdict = evaluate_target_pose(
            free_space,
            target,
            location,
            yaw,
            pitch_degrees,
            spec,
            non_occluders=non_occluders,
        )
        if not verdict["ok"]:
            result_rejects.append(
                {"reason": verdict["reason"], "location": location}
            )
            continue
        accepted.append(verdict["metrics"])
        used_azimuths.setdefault(target.entity_id, []).append(azimuth)
    accepted.sort(key=lambda row: (-float(row["score"]), row["location"]))
    z_values = sorted(float(row["location"][2]) for row in accepted)
    stats = {
        "attempts": attempts,
        "accepted": len(accepted),
        "kept": min(len(accepted), count),
        "acceptance_rate": (len(accepted) / attempts) if attempts else 0.0,
        "rejects": len(result_rejects),
        "reject_reasons": _count_reasons(result_rejects),
        "nodes": len(free_space.node_ids),
        "floor_nodes": 0,
        "floor_source": "room_shell" if room >= 0 else "none",
        "floor_z": floor_z,
        "height_policy": "entity_orbit",
        "targets": len(targets),
        "target_ids": [target.entity_id for target in targets],
        "targets_report": target_report,
        "rooms": free_space.container_node_ids,
        "camera_z_percentiles": _percentiles(z_values),
    }
    return {
        "accepted": accepted[:count],
        "rejects": result_rejects,
        "stats": stats,
        "targets": targets,
    }


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
    source_scene_digest: str = "",
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
            "strategy": str(spec.strategy),
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
            "ray_count": int(metrics.get("ray_count", 0)),
            "dominant_node_id": metrics.get("dominant_node_id"),
            "anchor_depth": float(metrics["anchor_depth"]),
            "depth_p10": float(metrics["depth_p10"]),
            "depth_p90": float(metrics["depth_p90"]),
            "room_node_id": metrics.get("room_node_id"),
            "container_node_ids": [
                str(value) for value in metrics.get("container_node_ids") or []
            ],
        },
        "lineage": {
            "blend_digest": base_scene_digest,
            "source_scene_digest": source_scene_digest or scene.get("source_digest"),
            "scene_camera_contract_version": SCENE_CAMERA_CONTRACT_VERSION,
            "camera_spec": spec.as_dict(),
            "scene_camera_digest": digest,
        },
    }
    target_entity_id = metrics.get("target_entity_id")
    if target_entity_id:
        # Entity-conditioned job: record what the sample is supposed to show.
        # ``canonical.origin`` is already the target centre and ``asset_size``
        # keeps its original meaning (camera-to-anchor distance), so the
        # renderer's light rig scaling stays correct without any change.
        label = str(metrics.get("target_label") or "")
        node_ids = [str(value) for value in metrics.get("target_node_ids") or []]
        job["camera"]["target_entity_id"] = str(target_entity_id)
        job["camera"]["target_label"] = label
        job["camera"]["target_visible_ratio"] = float(metrics["target_visible_ratio"])
        job["camera"]["target_screen_ratio"] = float(metrics["target_screen_ratio"])
        job["canonical"]["target_entity_id"] = str(target_entity_id)
        job["canonical"]["target_label"] = label
        job["canonical"]["target_radius"] = float(metrics["target_radius"])
        job["canonical"]["target_node_ids"] = node_ids
        job["free_space_report"]["occluder_node_ids"] = [
            str(value) for value in metrics.get("occluder_node_ids") or []
        ]
        job["lineage"]["target"] = {
            "entity_id": str(target_entity_id),
            "label": label,
            "category": str(metrics.get("target_category") or ""),
            "node_ids": node_ids,
            "center": [float(value) for value in metrics.get("target_center") or target],
            "radius": float(metrics["target_radius"]),
            "visible_ratio": float(metrics["target_visible_ratio"]),
            "screen_ratio": float(metrics["target_screen_ratio"]),
        }
    return job


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Content digest of a file, in the same ``sha256:<hex>`` form as io_utils."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def camera_count_split(count: int, mix_with_free_space: float) -> tuple[int, int]:
    """Split ``count`` cameras into ``(targeted, free_space)`` wide shots."""
    if count < 1:
        raise ValueError("count must be positive")
    mix = min(1.0, max(0.0, float(mix_with_free_space)))
    if count == 1:
        return 1, 0
    free = int(round(count * mix))
    free = min(free, count - 1)
    targeted = count - free
    return targeted, free


def _sample_scene_cameras(
    scene: dict[str, Any],
    spec: CameraSpec,
    *,
    seed: int,
    count: int,
) -> dict[str, Any]:
    """Dispatch on ``spec.strategy`` and merge both halves into one result."""
    if spec.strategy != "entity_conditioned_v1":
        return sample_camera_poses(scene, spec, seed=seed, count=count)
    targeted_count, free_count = camera_count_split(count, spec.mix_with_free_space)
    targeted = sample_targeted_poses(scene, spec, seed=seed, count=targeted_count)
    free: dict[str, Any] = {
        "accepted": [],
        "rejects": [],
        "stats": {
            "attempts": 0,
            "accepted": 0,
            "kept": 0,
            "acceptance_rate": 0.0,
            "rejects": 0,
            "reject_reasons": {},
            "nodes": 0,
            "floor_nodes": 0,
            "height_policy": "free_space",
        },
    }
    if free_count > 0 and targeted["targets"]:
        free = sample_camera_poses(
            scene,
            spec,
            seed=seed,
            count=free_count,
            targets=targeted["targets"],
            min_target_screen_ratio=spec.target.min_target_screen_ratio,
        )
    accepted = list(targeted["accepted"]) + list(free["accepted"])
    accepted.sort(key=lambda row: (-float(row["score"]), row["location"]))
    reasons = dict(targeted["stats"]["reject_reasons"])
    for reason, hits in free["stats"]["reject_reasons"].items():
        reasons[reason] = reasons.get(reason, 0) + int(hits)
    attempts = targeted["stats"]["attempts"] + free["stats"]["attempts"]
    rejects = targeted["stats"]["rejects"] + free["stats"]["rejects"]
    stats = {
        "attempts": attempts,
        "accepted": len(accepted),
        "kept": min(len(accepted), count),
        "acceptance_rate": (len(accepted) / attempts) if attempts else 0.0,
        "rejects": rejects,
        "reject_reasons": dict(sorted(reasons.items())),
        "nodes": max(targeted["stats"]["nodes"], free["stats"]["nodes"]),
        "floor_nodes": free["stats"]["floor_nodes"],
        "height_policy": "entity_conditioned_v1",
        "strategy": str(spec.strategy),
        "targeted_requested": targeted_count,
        "free_space_requested": free_count,
        "targets": targeted["stats"]["targets"],
        "target_ids": targeted["stats"]["target_ids"],
        "targeted_accepted": len(targeted["accepted"]),
        "free_space_accepted": len(free["accepted"]),
        # Diagnostics: why a scene produced (or failed to produce) subjects.
        "rooms": targeted["stats"].get("rooms") or free["stats"].get("rooms") or [],
        "floor_source": targeted["stats"].get("floor_source")
        or free["stats"].get("floor_source"),
        "floor_z": targeted["stats"].get("floor_z")
        if targeted["stats"].get("floor_z") is not None
        else free["stats"].get("floor_z"),
        "targets_report": targeted["stats"].get("targets_report") or {},
        "camera_z_percentiles": free["stats"].get("camera_z_percentiles")
        or targeted["stats"].get("camera_z_percentiles"),
    }
    return {
        "accepted": accepted[:count],
        "rejects": list(targeted["rejects"]) + list(free["rejects"]),
        "stats": stats,
        "targets": targeted["targets"],
    }


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
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Plan camera jobs for every scene in a ``scene.json`` document.

    ``scene.json`` stores ``normalized_blend`` relative to the project root, so
    ``project_root`` must be supplied to resolve and hash it; falling back to the
    process CWD would record a digest for the wrong file when the planner is
    launched from another directory.
    """
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
    root = (Path(project_root).expanduser() if project_root is not None else Path.cwd()).resolve()
    jobs: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    per_scene: dict[str, Any] = {}
    blend_digests: dict[Path, str] = {}
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        blend = blend_by_scene.get(scene_id, str(scene.get("normalized_blend", "")))
        # `base_scene_digest` must describe the *blend file* the renderer opens:
        # the renderer hashes that file and refuses stale jobs.  `source_digest`
        # is the source FBX and belongs in lineage only.
        if not blend:
            rejects.append({"scene_id": scene_id, "reason": "missing_normalized_blend_path"})
            continue
        blend_path = Path(blend).expanduser()
        if not blend_path.is_absolute():
            blend_path = (root / blend_path).resolve()
        if not blend_path.is_file():
            rejects.append(
                {
                    "scene_id": scene_id,
                    "reason": "missing_base_scene_blend",
                    "detail": str(blend_path),
                }
            )
            continue
        if blend_path not in blend_digests:
            blend_digests[blend_path] = _sha256_file(blend_path)
        digest = blend_digests[blend_path]
        source_digest = str(scene.get("source_digest", ""))
        fixture_ids = sorted(
            entity["entity_id"]
            for entity in scene.get("entities", [])
            if str(entity.get("category", "")) in fixtures
        )
        try:
            search = _sample_scene_cameras(
                scene, spec, seed=seed, count=cameras_per_scene
            )
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
                    source_scene_digest=source_digest,
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
    "ContainerSpec",
    "FreeSpace",
    "build_camera_job",
    "build_scene_camera_jobs",
    "blender_rotation_euler",
    "camera_basis",
    "camera_count_split",
    "detect_containers",
    "evaluate_pose",
    "evaluate_target_pose",
    "load_scene_document",
    "sample_camera_poses",
    "sample_targeted_poses",
    "target_screen_metrics",
    "view_directions",
]
