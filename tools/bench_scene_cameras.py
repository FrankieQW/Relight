"""Benchmark the scene-only camera search on a Bistro-scale synthetic scene.

Run from the parent directory of ``datacompose``::

    python -m datacompose.tools.bench_scene_cameras

No Blender, no data files: the scene is a deterministic random box soup whose
node count and extent are chosen to be in the same order of magnitude as one
normalised Bistro fragment.  The numbers it prints are what the feasibility note
quotes for camera-search cost per scene.
"""

from __future__ import annotations

import argparse
import random
import time

from datacompose.scene_cameras import (
    CameraSpec,
    FreeSpace,
    ProbeFreeSpace,
    _sample_scene_cameras,
    evaluate_pose,
)


def synthetic_scene(node_count: int = 4000, *, size: float = 200.0, seed: int = 20260810) -> dict:
    rng = random.Random(seed)
    nodes = []
    for index in range(node_count):
        x = rng.uniform(-size * 0.5, size * 0.5)
        y = rng.uniform(-size * 0.5, size * 0.5)
        z = rng.uniform(0.0, 18.0)
        extent = [rng.uniform(0.2, 4.0) for _ in range(3)]
        nodes.append(
            {
                "node_id": f"node_{index:05d}",
                "aabb_world": {
                    "min": [x, y, z],
                    "max": [x + extent[0], y + extent[1], z + extent[2]],
                },
                "triangle_count": 12,
            }
        )
    # A grid of floor slabs so the eye-height policy has something to stand on.
    entities = []
    step = 8.0
    floor_index = 0
    for grid_x in range(int(-size * 0.5), int(size * 0.5), int(step)):
        for grid_y in range(int(-size * 0.5), int(size * 0.5), int(step)):
            node_id = f"floor_{floor_index:04d}"
            nodes.append(
                {
                    "node_id": node_id,
                    "aabb_world": {
                        "min": [grid_x, grid_y, 0.0],
                        "max": [grid_x + step, grid_y + step, 0.05],
                    },
                    "triangle_count": 12,
                }
            )
            entities.append(
                {
                    "entity_id": f"scene:entity:{node_id}",
                    "node_ids": [node_id],
                    "raw_label": node_id,
                    "category": "floor",
                    "node_count": 1,
                }
            )
            floor_index += 1
    # A sparse set of framable props so the entity-conditioned strategy has
    # something to orbit; every other node stays anonymous geometry.
    for index in range(0, node_count, 40):
        node_id = f"node_{index:05d}"
        entities.append(
            {
                "entity_id": f"scene:entity:prop_{index:05d}",
                "node_ids": [node_id],
                "raw_label": f"Prop_{index:05d}",
                "category": "prop",
                "node_count": 1,
            }
        )
    return {
        "scene_id": "synthetic_bistro_like",
        "source_digest": "sha256:" + "0" * 64,
        "normalized_blend": "data/cache/scenes/synthetic.blend",
        "nodes": nodes,
        "entities": entities,
    }


def _hall_probe(scene: dict, *, resolution: float):
    """A clearance field for the scene bounding box, for query-cost timing.

    This is deliberately *not* a bake: it is the distance to the six faces of
    the scene bbox, so the only thing it measures is how expensive the
    probe-backed query path is per pose.  Cheaper than a real bake by orders of
    magnitude, and enough to answer "does the probe path cost more per pose than
    the AABB path?".
    """
    import numpy as np

    from datacompose.probes import ProbeVolume

    lows = np.array(
        [node["aabb_world"]["min"] for node in scene["nodes"] if node.get("aabb_world")],
        dtype=np.float64,
    )
    highs = np.array(
        [node["aabb_world"]["max"] for node in scene["nodes"] if node.get("aabb_world")],
        dtype=np.float64,
    )
    low = lows.min(axis=0)
    high = highs.max(axis=0)
    dims = tuple(int(value) for value in np.ceil((high - low) / resolution))
    axes = [
        low[axis] + (np.arange(dims[axis], dtype=np.float64) + 0.5) * resolution
        for axis in range(3)
    ]
    field = np.full(dims, np.inf)
    for axis in range(3):
        shaped = axes[axis].reshape(
            [-1 if index == axis else 1 for index in range(3)]
        )
        field = np.minimum(field, shaped - low[axis])
        field = np.minimum(field, high[axis] - shaped)
    cap = 1.0
    return ProbeVolume(
        origin=low,
        resolution=resolution,
        clearance=np.minimum(field, cap).astype(np.float32),
        nearest_node=np.zeros(dims, dtype=np.int32),
        node_ids=["walls"],
        clearance_cap=cap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--ray-grid", type=int, default=24)
    parser.add_argument(
        "--strategy",
        default="scene_free_space_v1",
        choices=["scene_free_space_v1", "entity_conditioned_v1"],
        help="camera planner to benchmark",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also time the probe-backed query path (query cost only, not a bake)",
    )
    parser.add_argument("--probe-resolution", type=float, default=1.0)
    args = parser.parse_args()

    scene = synthetic_scene(args.nodes)
    spec = CameraSpec(
        ray_grid=args.ray_grid,
        attempts_per_camera=args.attempts,
        strategy=args.strategy,
    )

    start = time.perf_counter()
    space = FreeSpace.from_scene(scene)
    build_seconds = time.perf_counter() - start
    low, high = space.bbox
    print(f"nodes={len(space.node_ids)} entities={len(scene['entities'])} strategy={spec.strategy}")
    print(f"bbox_min={[round(v, 2) for v in low]} bbox_max={[round(v, 2) for v in high]}")
    print(f"free_space_build={build_seconds * 1000:.1f} ms")

    start = time.perf_counter()
    for _ in range(200):
        evaluate_pose(space, [0.0, 0.0, 1.6], 37.0, -5.0, spec)
    per_pose = (time.perf_counter() - start) / 200.0
    print(f"pose_evaluation={per_pose * 1000:.2f} ms  (ray_grid={args.ray_grid} -> {args.ray_grid ** 2} rays)")

    if args.probe:
        start = time.perf_counter()
        probe = _hall_probe(scene, resolution=args.probe_resolution)
        hall_seconds = time.perf_counter() - start
        oracle = ProbeFreeSpace(probe, space, hit_clearance=0.10, step_scale=0.5)
        start = time.perf_counter()
        for _ in range(200):
            evaluate_pose(oracle, [0.0, 0.0, 1.6], 37.0, -5.0, spec)
        probe_pose = (time.perf_counter() - start) / 200.0
        start = time.perf_counter()
        for _ in range(200):
            oracle.sample_position(random.Random(0), spec)
        probe_sample = (time.perf_counter() - start) / 200.0
        print(
            f"probe: voxels={probe.voxel_count} resolution={probe.resolution} "
            f"field_build={hall_seconds:.2f} s (analytic hall, not a bake)"
        )
        print(
            f"probe_pose_evaluation={probe_pose * 1000:.2f} ms "
            f"probe_position_sample={probe_sample * 1000:.2f} ms "
            f"(same {args.ray_grid ** 2} rays)"
        )

    start = time.perf_counter()
    result = _sample_scene_cameras(scene, spec, seed=1, count=args.cameras)
    search_seconds = time.perf_counter() - start
    stats = result["stats"]
    print(
        f"search: cameras={len(result['accepted'])} attempts={stats['attempts']} "
        f"rate={stats['acceptance_rate']:.3f} total={search_seconds:.2f} s "
        f"({search_seconds / max(len(result['accepted']), 1):.2f} s per accepted camera)"
    )
    print(f"reject_reasons={stats['reject_reasons']}")
    for row in result["accepted"][:3]:
        print(
            "pose", [round(v, 2) for v in row["location"]],
            f"yaw={row['yaw_degrees']:.1f} pitch={row['pitch_degrees']:.1f}",
            f"score={row['score']:.3f} min_depth={row['minimum_hit_depth']:.2f}",
            f"spread={row['depth_spread_ratio']:.2f}",
            f"target={row.get('target_label')}",
            f"visible={row.get('target_visible_ratio', 0.0):.2f}",
            f"screen={row.get('target_screen_ratio', 0.0):.3f}",
        )


if __name__ == "__main__":
    main()
