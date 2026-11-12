from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import coordinates  # noqa: E402
from datacompose.scene_cameras import (  # noqa: E402
    CameraSpec,
    FreeSpace,
    build_camera_job,
    build_scene_camera_jobs,
    camera_basis,
    evaluate_pose,
    sample_camera_poses,
)


def _box(node_id: str, low, high) -> dict:
    return {
        "node_id": node_id,
        "raw_import_name": node_id,
        "source_fbx_element_id": None,
        "object_type": "MESH",
        "transform_world": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "aabb_world": {"min": list(low), "max": list(high)},
        "obb_world": {},
        "materials": [],
        "triangle_count": 12,
        "vertex_count": 8,
    }


def room_scene(scene_id: str = "room_a") -> dict:
    """6x6x3 m closed room with a 1.2x1.2x0.75 m table in the middle."""
    return {
        "scene_id": scene_id,
        "source_fbx": f"{scene_id}.fbx",
        "source_digest": "sha256:" + "0" * 64,
        "normalized_blend": f"data/cache/scenes/{scene_id}.blend",
        "units": "meter",
        "up_axis": "+Z",
        "config_digest": "sha256:" + "1" * 64,
        "nodes": [
            _box("floor", (-3.0, -3.0, -0.05), (3.0, 3.0, 0.0)),
            _box("ceiling", (-3.0, -3.0, 2.95), (3.0, 3.0, 3.0)),
            _box("wall_x_neg", (-3.05, -3.0, 0.0), (-3.0, 3.0, 3.0)),
            _box("wall_x_pos", (3.0, -3.0, 0.0), (3.05, 3.0, 3.0)),
            _box("wall_y_neg", (-3.0, -3.05, 0.0), (3.0, -3.0, 3.0)),
            _box("wall_y_pos", (-3.0, 3.0, 0.0), (3.0, 3.05, 3.0)),
            _box("table", (-0.6, -0.6, 0.0), (0.6, 0.6, 0.75)),
        ],
        "entities": [
            {
                "entity_id": "room_a:entity:floor",
                "node_ids": ["floor"],
                "raw_label": "Floor",
                "category": "floor",
                "category_confidence": 1.0,
                "grouping_confidence": 1.0,
                "grouping_method": "single_node",
                "replaceable": False,
                "support_surface": True,
                "obb_world": {},
            },
            {
                "entity_id": "room_a:entity:table",
                "node_ids": ["table"],
                "raw_label": "Table",
                "category": "table",
                "category_confidence": 1.0,
                "grouping_confidence": 1.0,
                "grouping_method": "single_node",
                "replaceable": True,
                "support_surface": True,
                "obb_world": {},
            },
        ],
        "stats": {"nodes": 7, "entities": 2},
    }


class FreeSpaceTest(unittest.TestCase):
    def test_point_clearance_and_inside_test(self) -> None:
        space = FreeSpace.from_scene(room_scene())
        inside_table = space.clearance([[0.0, 0.0, 0.4]])[0]
        self.assertEqual(inside_table, 0.0)
        middle_of_room = space.clearance([[0.0, 0.0, 1.5]])[0]
        self.assertGreater(middle_of_room, 0.5)
        self.assertTrue(bool(space.inside([[0.0, 0.0, 0.4]])[space.node_ids.index("table")]))

    def test_raycast_measures_distance_to_first_surface(self) -> None:
        space = FreeSpace.from_scene(room_scene())
        # Looking straight at the +X wall from the room centre.
        depth, index = space.raycast([0.0, 0.0, 1.5], [[1.0, 0.0, 0.0]], max_distance=45.0)
        self.assertAlmostEqual(float(depth[0]), 3.0, places=3)
        self.assertEqual(space.node_ids[int(index[0])], "wall_x_pos")
        # Looking at the table edge-on at table height.
        depth, index = space.raycast([-2.0, 0.0, 0.4], [[1.0, 0.0, 0.0]], max_distance=45.0)
        self.assertAlmostEqual(float(depth[0]), 1.4, places=3)
        self.assertEqual(space.node_ids[int(index[0])], "table")


class PoseEvaluationTest(unittest.TestCase):
    spec = CameraSpec()

    def setUp(self) -> None:
        self.space = FreeSpace.from_scene(room_scene())

    def test_camera_inside_object_is_rejected(self) -> None:
        verdict = evaluate_pose(self.space, [0.0, 0.0, 0.4], 0.0, 0.0, self.spec)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "camera_inside_geometry")

    def test_camera_pressed_into_wall_is_rejected(self) -> None:
        # 0.25 m from the +X wall: outside the camera radius but inside the near gate.
        verdict = evaluate_pose(self.space, [2.75, 0.0, 1.5], 0.0, 0.0, self.spec)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "near_geometry_in_view")
        self.assertLess(verdict["metrics"]["minimum_hit_depth"], self.spec.near_clearance)

    def test_staring_at_one_surface_is_rejected(self) -> None:
        verdict = evaluate_pose(self.space, [0.0, 0.0, 1.5], 0.0, 75.0, self.spec)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "single_node_dominates")
        self.assertGreater(verdict["metrics"]["single_node_fraction"], 0.9)

    def test_open_room_corner_view_is_accepted(self) -> None:
        verdict = evaluate_pose(self.space, [-2.2, -2.2, 1.5], 45.0, -8.0, self.spec)
        self.assertTrue(verdict["ok"], verdict.get("reason"))
        metrics = verdict["metrics"]
        self.assertGreaterEqual(metrics["hit_fraction"], 0.9)
        self.assertGreaterEqual(metrics["minimum_hit_depth"], self.spec.near_clearance)
        self.assertGreaterEqual(metrics["depth_spread_ratio"], self.spec.min_depth_spread_ratio)
        self.assertGreater(metrics["score"], 0.0)

    def test_anchor_lies_along_forward_axis(self) -> None:
        verdict = evaluate_pose(self.space, [-2.2, -2.2, 1.5], 45.0, -8.0, self.spec)
        anchor = verdict["metrics"]["anchor"]
        location = verdict["metrics"]["location"]
        forward = verdict["metrics"]["forward"]
        depth = verdict["metrics"]["anchor_depth"]
        for axis in range(3):
            self.assertAlmostEqual(anchor[axis], location[axis] + forward[axis] * depth, places=9)


class SamplingTest(unittest.TestCase):
    def test_sampling_is_deterministic(self) -> None:
        scene = room_scene()
        first = sample_camera_poses(scene, CameraSpec(), seed=11, count=3)
        second = sample_camera_poses(scene, CameraSpec(), seed=11, count=3)
        self.assertEqual(first["accepted"], second["accepted"])
        self.assertEqual(first["stats"]["attempts"], second["stats"]["attempts"])
        other = sample_camera_poses(scene, CameraSpec(), seed=12, count=3)
        self.assertNotEqual(
            [row["location"] for row in first["accepted"]],
            [row["location"] for row in other["accepted"]],
        )

    def test_floor_policy_keeps_eye_height_and_reports_acceptance(self) -> None:
        scene = room_scene()
        result = sample_camera_poses(scene, CameraSpec(), seed=5, count=8)
        self.assertEqual(result["stats"]["height_policy"], "floor_standing")
        self.assertGreaterEqual(len(result["accepted"]), 1)
        for row in result["accepted"]:
            self.assertGreaterEqual(row["location"][2], 1.10 - 1e-9)
            self.assertLessEqual(row["location"][2], 1.80 + 1e-9)
        self.assertGreater(result["stats"]["acceptance_rate"], 0.02)
        print(
            "\n[room_a camera search] "
            f"attempts={result['stats']['attempts']} accepted={result['stats']['accepted']} "
            f"rate={result['stats']['acceptance_rate']:.3f} "
            f"rejects={result['stats']['reject_reasons']}"
        )

    def test_free_height_policy_without_floor_entities(self) -> None:
        scene = room_scene()
        for entity in scene["entities"]:
            entity["category"] = "table" if entity["category"] == "floor" else entity["category"]
        result = sample_camera_poses(scene, CameraSpec(), seed=5, count=2)
        self.assertEqual(result["stats"]["height_policy"], "free")
        self.assertGreaterEqual(len(result["accepted"]), 1)

    def test_cameras_stay_apart_when_spacing_is_requested(self) -> None:
        scene = room_scene()
        spec = CameraSpec(min_camera_spacing=0.75)
        result = sample_camera_poses(scene, spec, seed=3, count=4)
        locations = [row["location"] for row in result["accepted"]]
        for index, left in enumerate(locations):
            for right in locations[index + 1 :]:
                distance = math.dist(left, right)
                self.assertGreaterEqual(distance, spec.min_camera_spacing - 1e-9)


class ManifestContractTest(unittest.TestCase):
    def test_job_satisfies_coordinates_contract(self) -> None:
        scene = room_scene()
        spec = CameraSpec()
        result = sample_camera_poses(scene, spec, seed=7, count=1)
        job = build_camera_job(
            scene,
            result["accepted"][0],
            spec=spec,
            seed=7,
            index=0,
            base_scene_blend=scene["normalized_blend"],
            base_scene_digest=scene["source_digest"],
            lighting_profile={"name": "scene_native_v1"},
        )
        # canonical.origin == camera.target, no-roll rotation, pinhole intrinsics.
        frame = coordinates.CameraFrame.from_row(job)
        self.assertGreater(frame.distance, 0.0)
        intrinsics = coordinates.Intrinsics.from_camera(job["camera"])
        self.assertIsNotNone(intrinsics)
        self.assertEqual(intrinsics.width, 512)
        self.assertAlmostEqual(intrinsics.cx, 256.0, places=6)
        self.assertAlmostEqual(intrinsics.cy, 256.0, places=6)
        # The camera axis must match the forward vector produced by the sampler.
        forward = job["camera"]["forward"]
        for axis, expected in enumerate(frame.forward):
            self.assertAlmostEqual(expected, forward[axis], places=6)
        self.assertEqual(job["job_id"], self._rebuild(scene, spec, job)["job_id"])

    def _rebuild(self, scene: dict, spec: CameraSpec, job: dict) -> dict:
        result = sample_camera_poses(scene, spec, seed=7, count=1)
        return build_camera_job(
            scene,
            result["accepted"][0],
            spec=spec,
            seed=7,
            index=0,
            base_scene_blend=scene["normalized_blend"],
            base_scene_digest=scene["source_digest"],
            lighting_profile={"name": "scene_native_v1"},
        )

    def test_scene_document_yields_jobs_per_scene(self) -> None:
        document = {
            "schema_version": "1.0",
            "scenes": [room_scene("room_a"), room_scene("room_b")],
        }
        result = build_scene_camera_jobs(document, seed=1, cameras_per_scene=2)
        self.assertEqual(len(result["jobs"]), 4)
        self.assertEqual({job["scene_id"] for job in result["jobs"]}, {"room_a", "room_b"})
        self.assertTrue(all(job["camera"]["strategy"] == "scene_free_space_v1" for job in result["jobs"]))
        self.assertEqual(result["stats"]["scenes"], 2)
        for job in result["jobs"]:
            coordinates.CameraFrame.from_row(job)

    def test_basis_is_orthonormal(self) -> None:
        for yaw in (0.0, 37.0, 180.0, 271.5):
            for pitch in (-20.0, -5.0, 0.0, 15.0):
                forward, right, up = camera_basis(yaw, pitch)
                self.assertAlmostEqual(sum(value * value for value in forward), 1.0, places=9)
                self.assertAlmostEqual(sum(value * value for value in right), 1.0, places=9)
                self.assertAlmostEqual(sum(value * value for value in up), 1.0, places=9)
                self.assertAlmostEqual(sum(a * b for a, b in zip(forward, right)), 0.0, places=9)
                self.assertAlmostEqual(sum(a * b for a, b in zip(forward, up)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
