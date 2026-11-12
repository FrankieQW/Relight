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

from datacompose.scene_cameras import CameraSpec, FreeSpace, evaluate_pose, sample_camera_poses


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
                }
            )
            entities.append(
                {
                    "entity_id": f"scene:entity:{node_id}",
                    "node_ids": [node_id],
                    "category": "floor",
                    "node_count": 1,
                }
            )
            floor_index += 1
    return {
        "scene_id": "synthetic_bistro_like",
        "source_digest": "sha256:" + "0" * 64,
        "normalized_blend": "data/cache/scenes/synthetic.blend",
        "nodes": nodes,
        "entities": entities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=4000)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--ray-grid", type=int, default=24)
    args = parser.parse_args()

    scene = synthetic_scene(args.nodes)
    spec = CameraSpec(ray_grid=args.ray_grid, attempts_per_camera=args.attempts)

    start = time.perf_counter()
    space = FreeSpace.from_scene(scene)
    build_seconds = time.perf_counter() - start
    low, high = space.bbox
    print(f"nodes={len(space.node_ids)} floors={len(scene['entities'])}")
    print(f"bbox_min={[round(v, 2) for v in low]} bbox_max={[round(v, 2) for v in high]}")
    print(f"free_space_build={build_seconds * 1000:.1f} ms")

    start = time.perf_counter()
    for _ in range(200):
        evaluate_pose(space, [0.0, 0.0, 1.6], 37.0, -5.0, spec)
    per_pose = (time.perf_counter() - start) / 200.0
    print(f"pose_evaluation={per_pose * 1000:.2f} ms  (ray_grid={args.ray_grid} -> {args.ray_grid ** 2} rays)")

    start = time.perf_counter()
    result = sample_camera_poses(scene, spec, seed=1, count=args.cameras)
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
        )


if __name__ == "__main__":
    main()
