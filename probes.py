"""Distance-field probes: real surface information, baked once, queried for free.

The AABB free-space model in :mod:`scene_cameras` is a *broad phase*: it
overestimates every object (a box always contains its mesh) and cannot tell a
hollow room shell from a solid crate.  Two failure modes follow, and both waste
render slots: a pose that looks free under boxes can sit inside a wall or a
Bathtub-shaped nook, and a room modelled as one mesh either blocks everything or
needs the ``container_v1`` guess.

The fix, borrowed from how BlenderProc reasons about scenes, is to stop guessing
from boxes and *ask the geometry*:

* bake the scene once into a voxel **clearance field** - the exact distance from
  each voxel centre to the nearest triangle, plus which node owns that triangle;
* the planner then samples positions in verified free space and casts view rays
  against the same field, so "is this pose free" and "what does this ray hit"
  are answered from real surfaces, not rectangles;
* AABBs keep doing what they are good at - cheap candidate culling and the
  geometric target filter - but they never make the final call.

Design notes
------------
* This module is pure NumPy: it runs in Blender (to bake) and in the planner
  (to query) with no ``bpy`` dependency, so the whole query side is unit
  testable offline.
* **Clearance is symmetric.**  Distance to the nearest surface is the same
  whether a point is inside or outside a mesh, which is why no inside/outside
  parity test is needed: "at least ``camera_radius`` from any triangle" already
  excludes the geometry.  The only case it misses is a solid thicker than twice
  the radius, and the Blender worker re-validates every pose exactly before
  rendering.
* **Floor support is derived, not baked.**  A column of free voxels starts where
  the clearance first exceeds the threshold; the world height of that boundary
  is where the camera can stand.  That reproduces BlenderProc's
  ``position_is_above_object`` idea (cast down, find the support surface) using
  data the clearance field already contains, and it is what makes multi-storey
  fragments work: the run a candidate belongs to *is* its floor.
* **Queries are conservative.**  ``distance-to-surface`` is 1-Lipschitz, so
  looking up the nearest voxel and subtracting the offset to that voxel centre
  yields a valid *lower* bound.  The planner is therefore never told there is
  more free space than there is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROBE_FORMAT = "lc_probe_v1"
PROBE_SUFFIX = ".probe.npz"

#: npz keys, shared by the Blender baker and the planner loader.
KEY_FORMAT = "format"
KEY_ORIGIN = "origin"
KEY_RESOLUTION = "resolution"
KEY_CLEARANCE = "clearance"
KEY_NEAREST_NODE = "nearest_node"
KEY_NODE_IDS = "node_ids"


def probe_path_for(blend_path: str | Path) -> Path:
    """``<stem>.probe.npz`` next to the ``.blend`` the probe was baked from."""
    return Path(blend_path).with_suffix(PROBE_SUFFIX)


#: Rough working-set target for one march block, in sampled points.
_BLOCK_POINTS = 64_000


def _block_steps(rays: int) -> int:
    """How many march steps fit in one vectorised block for ``rays`` rays."""
    return max(1, int(_BLOCK_POINTS // max(int(rays), 1)))


@dataclass(frozen=True)
class ProbeSpec:
    """Bake settings for the Blender-side probe stage (``probe-scenes``).

    Every value has a default *and* appears in ``configs/default.yaml`` under
    ``m4.probe``, so the resolution/cost trade-off is tunable without code.
    """

    enabled: bool = True
    #: Voxel edge length in metres. 10 cm resolves a chair leg and a door frame.
    resolution: float = 0.10
    #: Hard cap on ``nx * ny * nz``; the resolution is coarsened automatically
    #: until the grid fits. A 20 m building at 10 cm is 8 M voxels, so this is
    #: what keeps a bake at tens of seconds instead of minutes.
    max_voxels: int = 1_500_000
    #: Distances are clipped here: nothing in the planner needs to know that a
    #: point is 20 m from a wall, only that it is not 0.2 m from one.
    clearance_cap: float = 1.00
    #: Extra margin around the scene bounding box, in metres.
    padding: float = 0.10
    workers: int = 1
    resume: bool = True
    #: ``true`` makes the stage fail loudly on the first scene that cannot be
    #: baked; ``false`` (default) records the failure and lets planning fall
    #: back to AABBs for that scene only.
    require_all: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "resolution": float(self.resolution),
            "max_voxels": int(self.max_voxels),
            "clearance_cap": float(self.clearance_cap),
            "padding": float(self.padding),
            "workers": int(self.workers),
            "resume": bool(self.resume),
            "require_all": bool(self.require_all),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ProbeSpec":
        source: Mapping[str, Any] = value or {}
        defaults = cls()
        resolution = float(source.get("resolution", defaults.resolution))
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("m4.probe.resolution must be a positive number")
        return cls(
            enabled=bool(source.get("enabled", defaults.enabled)),
            resolution=resolution,
            max_voxels=int(source.get("max_voxels", defaults.max_voxels)),
            clearance_cap=float(source.get("clearance_cap", defaults.clearance_cap)),
            padding=float(source.get("padding", defaults.padding)),
            workers=int(source.get("workers", defaults.workers)),
            resume=bool(source.get("resume", defaults.resume)),
            require_all=bool(source.get("require_all", defaults.require_all)),
        )


class ProbeVolume:
    """A baked clearance field plus the node that owns each nearest triangle.

    ``clearance`` and ``nearest_node`` are ``(nx, ny, nz)`` arrays indexed by
    ``floor((point - origin) / resolution)``; ``node_ids`` maps
    ``nearest_node`` values to the node identity the baker recorded.
    """

    def __init__(
        self,
        *,
        origin: Sequence[float],
        resolution: float,
        clearance: np.ndarray,
        nearest_node: np.ndarray,
        node_ids: Sequence[str],
        clearance_cap: float | None = None,
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.resolution = float(resolution)
        if self.resolution <= 0.0:
            raise ValueError("probe resolution must be positive")
        self.clearance = np.ascontiguousarray(clearance, dtype=np.float32)
        if self.clearance.ndim != 3:
            raise ValueError("probe clearance must be a 3-D array")
        self.nearest_node = np.ascontiguousarray(nearest_node, dtype=np.int32)
        if self.nearest_node.shape != self.clearance.shape:
            raise ValueError("probe clearance and nearest_node must have equal shape")
        self.node_ids = [str(value) for value in node_ids]
        self.clearance_cap = float(
            clearance_cap if clearance_cap is not None else float(self.clearance.max())
        )
        self._cache: dict[tuple[Any, ...], Any] = {}

    # -- metadata ----------------------------------------------------------
    @property
    def dims(self) -> tuple[int, int, int]:
        return (
            int(self.clearance.shape[0]),
            int(self.clearance.shape[1]),
            int(self.clearance.shape[2]),
        )

    @property
    def voxel_count(self) -> int:
        return int(self.clearance.size)

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        low = [float(value) for value in self.origin]
        high = [
            float(self.origin[axis] + self.dims[axis] * self.resolution)
            for axis in range(3)
        ]
        return low, high

    def summary(self) -> dict[str, Any]:
        low, high = self.bounds
        return {
            "format": PROBE_FORMAT,
            "resolution": self.resolution,
            "dims": list(self.dims),
            "voxels": self.voxel_count,
            "nodes": len(self.node_ids),
            "bounds": {"min": low, "max": high},
            "clearance_cap": self.clearance_cap,
        }

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            **{
                KEY_FORMAT: np.array(PROBE_FORMAT),
                KEY_ORIGIN: self.origin.astype(np.float64),
                KEY_RESOLUTION: np.array(self.resolution, dtype=np.float64),
                KEY_CLEARANCE: self.clearance,
                KEY_NEAREST_NODE: self.nearest_node,
                KEY_NODE_IDS: np.array(self.node_ids, dtype=np.str_),
            },
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ProbeVolume":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            fmt = str(payload[KEY_FORMAT])
            if fmt != PROBE_FORMAT:
                raise ValueError(
                    f"unsupported probe format {fmt!r} in {source} "
                    f"(expected {PROBE_FORMAT!r})"
                )
            return cls(
                origin=payload[KEY_ORIGIN],
                resolution=float(payload[KEY_RESOLUTION]),
                clearance=payload[KEY_CLEARANCE],
                nearest_node=payload[KEY_NEAREST_NODE],
                node_ids=[str(value) for value in payload[KEY_NODE_IDS]],
                clearance_cap=float(payload[KEY_CLEARANCE].max()),
            )

    # -- lookup ------------------------------------------------------------
    def _lookup(
        self, points: Sequence[Sequence[float]] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest voxel per point: ``(flat_index, offset, valid)``.

        ``offset`` is the distance from the query point to that voxel centre and
        is what makes the clearance query a guaranteed lower bound.
        """
        array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        relative = (array - self.origin[None, :]) / self.resolution - 0.5
        indices = np.floor(relative + 0.5).astype(np.int64)
        dims = np.asarray(self.dims, dtype=np.int64)[None, :]
        valid = np.all((indices >= 0) & (indices < dims), axis=1)
        np.clip(indices, 0, dims - 1, out=indices)
        centres = self.origin[None, :] + (indices + 0.5) * self.resolution
        offset = np.linalg.norm(array - centres, axis=1)
        flat = (
            indices[:, 0] * (self.dims[1] * self.dims[2])
            + indices[:, 1] * self.dims[2]
            + indices[:, 2]
        )
        return flat, offset, valid

    def clearance_at(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Conservative lower bound on the distance to the nearest surface."""
        flat, offset, valid = self._lookup(points)
        values = self.clearance.reshape(-1)[flat].astype(np.float64) - offset
        return np.where(valid, np.maximum(values, 0.0), self.clearance_cap)

    def nearest_node_at(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        flat, _offset, valid = self._lookup(points)
        values = self.nearest_node.reshape(-1)[flat]
        return np.where(valid, values, np.int32(-1)).astype(np.int32)

    def inside(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        _flat, _offset, valid = self._lookup(points)
        return valid

    # -- masks -------------------------------------------------------------
    def free_mask(self, clearance: float) -> np.ndarray:
        """Voxels whose centre is at least ``clearance`` away from any surface."""
        key = ("free", round(float(clearance), 6))
        cached = self._cache.get(key)
        if cached is None:
            cached = self.clearance >= float(clearance)
            self._cache[key] = cached
        return cached

    def run_base(self, clearance: float) -> np.ndarray:
        """Per voxel: bottom slice index of the free run it belongs to (``-1``).

        A column of free voxels begins where the clearance first exceeds the
        threshold, so the run a voxel belongs to *is* its floor - which is how a
        multi-storey fragment keeps its cameras on the right storey.
        """
        key = ("base", round(float(clearance), 6))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        free = self.free_mask(clearance)
        starts = free.copy()
        starts[:, :, 1:] &= ~free[:, :, :-1]
        z_index = np.arange(free.shape[2], dtype=np.int32)[None, None, :]
        marker = np.where(starts, z_index, np.int32(-1))
        base = np.maximum.accumulate(marker, axis=2)
        self._cache[key] = base
        return base

    def support_z(self, clearance: float) -> np.ndarray:
        """World height of the surface directly below each voxel (``nan`` if none).

        ``clearance`` measured at the bottom voxel of a run is how far that voxel
        floats above the surface, so subtracting it turns a voxel centre into a
        surface height.  The cap stops a corner column (nearest surface is a
        wall, not the floor) from pushing the estimate metres downwards.
        """
        key = ("support", round(float(clearance), 6))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = self.run_base(clearance)
        has_floor = base > 0
        safe = np.where(has_floor, base, 0)
        centre_z = self.origin[2] + (safe + 0.5) * self.resolution
        boundary = np.take_along_axis(self.clearance.astype(np.float64), safe, axis=2)
        correction = np.minimum(boundary, float(clearance) + self.resolution)
        floor_z = np.where(has_floor, centre_z - correction, np.nan)
        self._cache[key] = floor_z
        return floor_z

    def height_above_support(self, clearance: float) -> np.ndarray:
        """Per voxel: height above the surface directly below it (``nan`` if none)."""
        centre_z = self.origin[2] + (
            np.arange(self.dims[2], dtype=np.float64) + 0.5
        ) * self.resolution
        return centre_z[None, None, :] - self.support_z(clearance)

    # -- sampling ----------------------------------------------------------
    def candidate_mask(
        self,
        *,
        clearance: float,
        eye_height: tuple[float, float] | None = None,
        height_tolerance: float = 0.0,
    ) -> np.ndarray:
        """Voxels that are both free *and* at a plausible standing height."""
        mask = self.free_mask(clearance)
        if eye_height is None:
            return mask
        low = float(min(eye_height)) - float(height_tolerance)
        high = float(max(eye_height)) + float(height_tolerance)
        height = self.height_above_support(clearance)
        with np.errstate(invalid="ignore"):
            band = (height >= low) & (height <= high)
        return mask & band

    def candidate_positions(
        self,
        *,
        clearance: float,
        eye_height: tuple[float, float] | None = None,
        height_tolerance: float = 0.0,
    ) -> np.ndarray:
        """World-space centres of every standable free voxel, shape ``(N, 3)``."""
        key = (
            "positions",
            round(float(clearance), 6),
            None if eye_height is None else tuple(sorted(float(v) for v in eye_height)),
            round(float(height_tolerance), 6),
        )
        cached = self._cache.get(key)
        if cached is None:
            mask = self.candidate_mask(
                clearance=clearance,
                eye_height=eye_height,
                height_tolerance=height_tolerance,
            )
            indices = np.argwhere(mask)
            if len(indices):
                cached = self.origin[None, :] + (indices + 0.5) * self.resolution
            else:
                cached = np.zeros((0, 3), dtype=np.float64)
            self._cache[key] = cached
        return cached

    def random_position(
        self,
        rng: Any,
        *,
        clearance: float,
        eye_height: tuple[float, float] | None = None,
        height_tolerance: float = 0.0,
    ) -> list[float] | None:
        """One standable free position, or ``None`` when the scene has none."""
        pool = self.candidate_positions(
            clearance=clearance,
            eye_height=eye_height,
            height_tolerance=height_tolerance,
        )
        if not len(pool):
            return None
        return [float(value) for value in pool[rng.randrange(len(pool))]]

    # -- ray casting -------------------------------------------------------
    def cast(
        self,
        origin: Sequence[float],
        directions: np.ndarray,
        *,
        max_distance: float,
        threshold: float,
        step_scale: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """First surface along each ray, as ``(depth, node_index)``.

        A ray "hits" where the clearance drops below ``threshold``, which is the
        discrete analogue of the exact ``BVHTree`` hit in
        ``blender/render_scene.py``.  ``node_index`` is the bake-local index into
        :attr:`node_ids` (``-1`` when the ray leaves the probe without a hit).

        Sampling is a fixed-step march rather than a DDA walk: it is fully
        vectorised over rays, and with ``step <= threshold`` a surface cannot be
        tunnelled through.
        """
        dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        count = len(dirs)
        depth = np.full(count, -1.0, dtype=np.float64)
        nodes = np.full(count, -1, dtype=np.int32)
        if not count or max_distance <= 0.0:
            return depth, nodes
        step = max(self.resolution * float(step_scale), 1e-3)
        steps = max(1, int(math.ceil(max_distance / step)))
        origin_row = np.asarray(origin, dtype=np.float64).reshape(3)
        distances = np.arange(1, steps + 1, dtype=np.float64) * step
        flat_clearance = self.clearance.reshape(-1)
        flat_node = self.nearest_node.reshape(-1)
        # March in step-blocks and drop rays as soon as they hit: interior views
        # resolve within a few metres, so most of the 45 m budget is never
        # sampled.  The block size keeps the working set around a few hundred
        # thousand points whatever the ray count is.
        pending = np.arange(count)
        cursor = 0
        while cursor < steps and len(pending):
            block = max(1, min(steps - cursor, _block_steps(len(pending))))
            block_distances = distances[cursor : cursor + block]
            points = (
                origin_row[None, None, :]
                + block_distances[:, None, None]
                * dirs[pending][None, :, :]
            ).reshape(-1, 3)
            index, offset, valid = self._lookup(points)
            values = flat_clearance[index].astype(np.float64) - offset
            clear = (valid & (values < float(threshold))).reshape(block, len(pending))
            found = clear.any(axis=0)
            if bool(found.any()):
                first = np.argmax(clear, axis=0)
                picked = np.nonzero(found)[0]
                hit_nodes = flat_node[index].reshape(block, len(pending))
                depth[pending[picked]] = block_distances[first[picked]]
                nodes[pending[picked]] = hit_nodes[first[picked], picked]
                pending = pending[~found]
            cursor += block
        return depth, nodes


def discover_probes(
    scene_document: Mapping[str, Any],
    *,
    project_root: str | Path,
    suffix: str = PROBE_SUFFIX,
) -> dict[str, Path]:
    """Map ``scene_id`` to its baked probe, skipping scenes that have none.

    Probes live next to the ``.blend`` they were baked from, so the path is
    derived from ``scene.json`` rather than stored in it - the scene graph stays
    a description of the scene, not of the pipeline's caches.
    """
    root = Path(project_root)
    found: dict[str, Path] = {}
    for scene in scene_document.get("scenes", []) or []:
        blend = str(scene.get("normalized_blend") or "")
        if not blend:
            continue
        path = Path(blend).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        candidate = probe_path_for(path)
        if candidate.suffix != suffix and not candidate.name.endswith(suffix):
            continue
        if candidate.is_file():
            found[str(scene.get("scene_id"))] = candidate
    return found


def load_probes(paths: Mapping[str, str | Path]) -> dict[str, ProbeVolume]:
    """Load every probe that exists, ignoring the ones that do not."""
    loaded: dict[str, ProbeVolume] = {}
    for scene_id, path in paths.items():
        source = Path(path)
        if source.is_file():
            loaded[str(scene_id)] = ProbeVolume.load(source)
    return loaded


__all__ = [
    "PROBE_FORMAT",
    "PROBE_SUFFIX",
    "ProbeSpec",
    "ProbeVolume",
    "discover_probes",
    "load_probes",
    "probe_path_for",
]
