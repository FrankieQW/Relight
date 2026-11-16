"""Bake a scene into a triangle-accurate clearance probe (Blender side).

Run by ``scene_probe.py`` as::

    blender --background --factory-startup --python blender/probe_scene.py -- \
        probe --job /path/to/job.json

Why a separate stage
--------------------
The planner's AABB free-space model cannot see inside geometry: a box always
contains its mesh, so a room shell and a solid block look identical, and a
L-shaped room is a rectangle.  Every such approximation is a pose the planner
accepts and the renderer then rejects, which is exactly the "empty wall corner,
no object" symptom this pipeline was built to remove.

So the geometry is asked directly, once, and the answer is cached:

* one world-space ``BVHTree`` over every rendered triangle;
* per voxel, ``find_nearest`` gives the exact distance to the nearest triangle
  *and* the triangle's node - no inside/outside test needed, because distance to
  the nearest surface is the same on both sides of a wall;
* the planner then samples positions and casts view rays against those numbers.

Everything is defensive: a scene that cannot be baked writes no probe and the
planner silently falls back to AABBs for that scene, so this stage can never
break an existing dataset.

This file shares no code with BlenderProc (GPL-3.0); it only follows the same
*idea* of validating candidates against real geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
# Blender runs this file by path with its own Python, so register the package
# under its canonical name: the checkout folder may have any name.
from _launcher import ensure_package  # noqa: E402

ensure_package()

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

from datacompose.probes import ProbeVolume  # noqa: E402


#: Grow the voxel edge by this factor until the grid fits ``max_voxels``.
_COARSEN_FACTOR = 1.15


def main() -> None:
    arguments = _parse_args()
    if arguments.command == "probe":
        _probe(Path(arguments.job))
    else:  # pragma: no cover - argparse enforces the command value.
        raise ValueError(arguments.command)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="LightConstruction probe baker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--job", required=True)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Geometry collection
# --------------------------------------------------------------------------- #
def _renderable_meshes() -> list[Any]:
    """Every rendered mesh, in a stable order (node identity must be stable)."""
    objects = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    return sorted(objects, key=lambda item: (item.name, item.data.name))


def _node_id(obj) -> str:
    """The ``scene.json`` node id ``entry.py`` stamped on the object at import."""
    value = obj.get("lc_scene_object_id")
    if value:
        return str(value)
    return str(obj.name)


def _triangle_soup(objects):
    """World-space vertices, triangles and the node owner of each triangle.

    Triangles are taken from ``loop_triangles`` so every polygon is a real
    triangle (``BVHTree.FromPolygons(all_triangles=True)``); quads and ngons are
    fanned by Blender, which keeps the nearest-hit triangle index unambiguous.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    node_ids: list[str] = []
    offset = 0
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not len(mesh.vertices) or not len(mesh.polygons):
                continue
            mesh.calc_loop_triangles()
            if not len(mesh.loop_triangles):
                continue
            count = len(mesh.vertices)
            coordinates = np.empty(count * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", coordinates)
            coordinates = coordinates.reshape(count, 3)
            matrix = np.array(evaluated.matrix_world, dtype=np.float64)
            world = coordinates @ matrix[:3, :3].T + matrix[:3, 3]
            corner_count = len(mesh.loop_triangles) * 3
            corners = np.empty(corner_count, dtype=np.int64)
            mesh.loop_triangles.foreach_get("vertices", corners)
            corners = corners.reshape(-1, 3) + offset
        finally:
            evaluated.to_mesh_clear()
        index = len(node_ids)
        node_ids.append(_node_id(obj))
        vertices.append(world)
        triangles.append(corners)
        owners.append(np.full(len(corners), index, dtype=np.int32))
        offset += len(world)
    if not triangles:
        return None
    return (
        np.concatenate(vertices).tolist(),
        np.concatenate(triangles).tolist(),
        np.concatenate(owners),
        node_ids,
    )


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #
def _probe(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    started = time.time()
    bpy.ops.wm.open_mainfile(filepath=str(job["blend"]))
    bpy.context.view_layer.update()

    objects = _renderable_meshes()
    soup = _triangle_soup(objects)
    if soup is None:
        raise RuntimeError(f"{job['scene_id']}: no rendered triangles to bake")
    vertices, triangles, owners, node_ids = soup
    print(
        f"[probe] {job['scene_id']}: {len(vertices)} vertices, "
        f"{len(triangles)} triangles, {len(node_ids)} nodes",
        flush=True,
    )
    bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    if bvh is None:  # pragma: no cover - defensive, FromPolygons rarely returns None
        raise RuntimeError(f"{job['scene_id']}: BVH construction failed")

    low, high, resolution, dims = _grid(job, np.asarray(vertices))
    clearance, nearest = _sample(bvh, owners, low, resolution, dims, job)
    probe = ProbeVolume(
        origin=low,
        resolution=resolution,
        clearance=clearance,
        nearest_node=nearest,
        node_ids=node_ids,
        clearance_cap=float(job.get("clearance_cap", 1.0)),
    )
    path = probe.save(job["output"])
    print(
        f"[probe] {job['scene_id']}: {probe.voxel_count} voxels at "
        f"{resolution:.3f} m -> {path} ({time.time() - started:.1f}s)",
        flush=True,
    )
    summary = {
        "scene_id": job["scene_id"],
        "output": str(path),
        "dims": list(dims),
        "resolution": resolution,
        "voxels": probe.voxel_count,
        "triangles": len(triangles),
        "nodes": len(node_ids),
        "seconds": round(time.time() - started, 2),
    }
    Path(job["summary"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def _grid(job: dict[str, Any], vertices: np.ndarray):
    """Voxel origin, edge length and dimensions, coarsened to fit the budget."""
    padding = float(job.get("padding", 0.0) or 0.0)
    low = vertices.min(axis=0) - padding
    high = vertices.max(axis=0) + padding
    extent = np.maximum(high - low, 1e-6)
    resolution = float(job.get("resolution", 0.1))
    max_voxels = max(1, int(job.get("max_voxels", 1_500_000)))
    while True:
        dims = np.maximum(np.ceil(extent / resolution).astype(np.int64), 1)
        if int(np.prod(dims)) <= max_voxels:
            break
        # A whole-building bbox at 10 cm is tens of millions of voxels; growing
        # the edge is the only knob that keeps a bake in the tens of seconds.
        resolution *= _COARSEN_FACTOR
        if resolution > 5.0:  # pragma: no cover - absurd scenes only
            raise RuntimeError("cannot fit the voxel budget at a sane resolution")
    return low, high, float(resolution), tuple(int(value) for value in dims)


def _sample(bvh, owners: np.ndarray, origin: np.ndarray, resolution: float, dims, job):
    """Fill the clearance and nearest-node fields, one voxel at a time.

    ``bpy``'s BVH has no vectorised query, so this is a plain Python loop.  A
    36-triangle scene bakes at ~1.5 us per voxel (1.2 M voxels in 1.7 s);
    expect a few us per voxel once the tree holds a million triangles, because
    ``find_nearest`` cost grows with the tree.  The ``max_voxels`` budget in
    :func:`_grid` is what bounds the total.  Progress is printed so a long bake
    is not silent.
    """
    cap = float(job.get("clearance_cap", 1.0))
    nx, ny, nz = dims
    clearance = np.empty((nx, ny, nz), dtype=np.float32)
    nearest = np.full((nx, ny, nz), -1, dtype=np.int32)
    find_nearest = bvh.find_nearest
    x_values = origin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
    y_values = origin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * resolution
    z_values = origin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * resolution
    report = max(1, nx // 10)
    for ix in range(nx):
        x = float(x_values[ix])
        for iy in range(ny):
            y = float(y_values[iy])
            row_clearance = clearance[ix, iy]
            row_nearest = nearest[ix, iy]
            for iz in range(nz):
                hit = find_nearest((x, y, float(z_values[iz])))
                if hit is None:
                    row_clearance[iz] = cap
                    continue
                distance = float(hit[3])
                row_clearance[iz] = cap if distance > cap else distance
                index = int(hit[2])
                if 0 <= index < len(owners):
                    row_nearest[iz] = owners[index]
        if ix % report == 0:
            print(f"[probe] {job['scene_id']}: column {ix + 1}/{nx}", flush=True)
    return clearance, nearest


if __name__ == "__main__":
    main()
