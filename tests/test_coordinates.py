from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import coordinates  # noqa: E402


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "transfer.py"


def load_reference():
    """Load the bundled transfer.py that defines the reference convention."""
    path = REFERENCE_PATH
    spec = importlib.util.spec_from_file_location("transfer_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_aligned_row() -> dict:
    """Camera on -Y looking at the origin; a light at canonical [2, 3, 1]."""
    return {
        "canonical": {
            "origin": [0.0, 0.0, 0.0],
            "position_axes": "x=right,y=camera-forward,z=up",
            "units": "meter",
            "asset_size": 1.0,
        },
        "camera": {
            "location": [0.0, -10.0, 0.0],
            "target": [0.0, 0.0, 0.0],
            "rotation_euler": [math.pi / 2.0, 0.0, 0.0],
            "focal_length": 50.0,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "resolution": {"width": 1024, "height": 1024},
            "intrinsics": {
                "width": 1024,
                "height": 1024,
                "fx": 512.0,
                "fy": 512.0,
                "cx": 512.0,
                "cy": 512.0,
            },
        },
        "point_lights": [{"id": "light_0", "position": [2.0, 3.0, 1.0]}],
        "in_scene_lights": [],
    }


def rotated_row() -> dict:
    """No-roll camera with both azimuth and elevation, built from the Euler convention.

    The renderer aims the camera with ``to_track_quat("-Z", "Y")``, which never
    introduces roll.  ``rx = pi/2 - elevation`` with ``ry = 0`` reproduces that
    family exactly, so the rotation check accepts it.
    """
    elevation = 0.35
    azimuth = 2.1
    rx = math.pi / 2.0 - elevation
    ry = 0.0
    rz = azimuth
    sx, cx = math.sin(rx), math.cos(rx)  # cx = sin(elevation), sx = cos(elevation)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    forward = [
        -(cz * sy * cx + sz * sx),
        -(sz * sy * cx - cz * sx),
        -cy * cx,
    ]
    distance = 5.0
    target = [1.0, -2.0, 1.5]
    location = [t - distance * f for t, f in zip(target, forward)]
    return {
        "canonical": {
            "origin": target,
            "position_axes": "x=right,y=camera-forward,z=up",
            "units": "meter",
            "asset_size": 1.0,
        },
        "camera": {
            "location": location,
            "target": target,
            "rotation_euler": [rx, ry, rz],
            "focal_length": 35.0,
            "shift_x": 0.02,
            "shift_y": -0.01,
            "angle_x": math.radians(60.0),
            "angle_y": math.radians(60.0),
            "resolution": {"width": 960, "height": 960},
        },
        "point_lights": [{"id": "light_0", "position": [0.7, 1.4, 0.5]}],
        "in_scene_lights": [
            {"id": "fixture_0", "position": [0.0, 0.0, 0.0]},
        ],
    }


class ReferenceEquivalenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = load_reference()

    def test_camera_position_matches_transfer_py(self) -> None:
        for name, row in (("axis", axis_aligned_row()), ("rotated", rotated_row())):
            with self.subTest(camera=name):
                expected, expected_lights = self.reference.convert_row(json.loads(json.dumps(row)))
                actual, actual_lights, intrinsics = coordinates.convert_row(
                    json.loads(json.dumps(row))
                )
                self.assertEqual(actual_lights, expected_lights)
                self.assertTrue(intrinsics)
                for field in ("point_lights", "in_scene_lights"):
                    for got, want in zip(actual[field], expected[field]):
                        self.assertEqual(got["camera_position"], want["camera_position"])
                        # The reference keeps the original canonical position.
                        self.assertEqual(got["position"], want["position"])

    def test_renderer_position_validation_agrees_with_transfer_py(self) -> None:
        row = axis_aligned_row()
        frame = coordinates.CameraFrame.from_row(row)
        camera_point = frame.canonical_to_camera([2.0, 3.0, 1.0])
        row["point_lights"][0]["renderer_position"] = frame.camera_to_world(camera_point)
        reference, _ = self.reference.convert_row(json.loads(json.dumps(row)))
        converted, _, _ = coordinates.convert_row(row)
        self.assertEqual(
            converted["point_lights"][0]["camera_position"],
            reference["point_lights"][0]["camera_position"],
        )

    def test_both_implementations_reject_a_mismatched_renderer_position(self) -> None:
        row = axis_aligned_row()
        row["point_lights"][0]["renderer_position"] = [9.0, 9.0, 9.0]
        with self.assertRaises(ValueError):
            self.reference.convert_row(json.loads(json.dumps(row)))
        with self.assertRaises(ValueError):
            coordinates.convert_row(json.loads(json.dumps(row)))


class CameraFrameTest(unittest.TestCase):
    def test_axis_aligned_conversion_and_depth(self) -> None:
        frame = coordinates.CameraFrame.from_row(axis_aligned_row())
        self.assertEqual(frame.distance, 10.0)
        self.assertEqual(frame.canonical_to_camera([2.0, 3.0, 1.0]), [2.0, -1.0, 13.0])
        self.assertEqual(frame.camera_to_canonical([2.0, -1.0, 13.0]), [2.0, 3.0, 1.0])

    def test_world_and_camera_round_trip(self) -> None:
        for row in (axis_aligned_row(), rotated_row()):
            frame = coordinates.CameraFrame.from_row(row)
            world = [0.3, -1.2, 4.4]
            camera_point = frame.world_to_camera(world)
            restored = frame.camera_to_world(camera_point)
            for got, want in zip(restored, world):
                self.assertAlmostEqual(got, want, places=12)

    def test_depth_is_axial_not_euclidean(self) -> None:
        frame = coordinates.CameraFrame.from_row(axis_aligned_row())
        camera_point = frame.canonical_to_camera([2.0, 3.0, 1.0])
        euclidean = math.dist(frame.location, frame.camera_to_world(camera_point))
        self.assertAlmostEqual(camera_point[2], 13.0, places=12)
        self.assertGreater(euclidean, camera_point[2])
        self.assertAlmostEqual(euclidean, math.sqrt(2.0**2 + (-1.0) ** 2 + 13.0**2), places=12)

    def test_rotation_that_does_not_face_the_target_is_rejected(self) -> None:
        row = axis_aligned_row()
        row["camera"]["rotation_euler"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            coordinates.CameraFrame.from_row(row)

    def test_origin_must_match_camera_target(self) -> None:
        row = axis_aligned_row()
        row["canonical"]["origin"] = [1.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            coordinates.CameraFrame.from_row(row)


class IntrinsicsTest(unittest.TestCase):
    def test_pinhole_projection_matches_the_stored_intrinsics(self) -> None:
        row, _, _ = coordinates.convert_row(axis_aligned_row())
        light = row["point_lights"][0]
        plane = light["image_plane"]
        x, y, depth = light["camera_position"]
        self.assertAlmostEqual(plane["u"], 512.0 * x / depth + 512.0, places=12)
        self.assertAlmostEqual(plane["v"], 512.0 * y / depth + 512.0, places=12)
        self.assertAlmostEqual(plane["x"], x / depth, places=12)
        self.assertAlmostEqual(plane["y"], y / depth, places=12)
        self.assertEqual(plane["depth"], 13.0)
        self.assertTrue(plane["in_front"])

    def test_normalized_image_plane_rebuilds_the_camera_vector(self) -> None:
        row, _, _ = coordinates.convert_row(rotated_row())
        for field in ("point_lights", "in_scene_lights"):
            for light in row[field]:
                plane = light["image_plane"]
                rebuilt = coordinates.image_plane_to_camera(
                    plane["x"], plane["y"], plane["depth"]
                )
                for got, want in zip(rebuilt, light["camera_position"]):
                    self.assertAlmostEqual(got, want, places=12)

    def test_intrinsics_from_fov_and_principal_point_shift(self) -> None:
        camera = {
            "angle_x": math.radians(90.0),
            "angle_y": math.radians(90.0),
            "shift_x": 0.1,
            "shift_y": -0.05,
            "resolution": {"width": 1000, "height": 800},
        }
        intrinsics = coordinates.Intrinsics.from_camera(camera)
        self.assertIsNotNone(intrinsics)
        self.assertAlmostEqual(intrinsics.fx, 500.0, places=9)
        self.assertAlmostEqual(intrinsics.fy, 400.0, places=9)
        self.assertAlmostEqual(intrinsics.cx, 400.0, places=9)
        self.assertAlmostEqual(intrinsics.cy, 360.0, places=9)

    def test_resolution_override_supplies_pixels_for_older_manifests(self) -> None:
        row = axis_aligned_row()
        row["camera"].pop("intrinsics")
        row["camera"].pop("resolution")
        with self.assertRaises(ValueError):
            coordinates.convert_row(json.loads(json.dumps(row)))
        converted, _, intrinsics = coordinates.convert_row(
            json.loads(json.dumps(row)), resolution=(1024, 1024), require_intrinsics=False
        )
        light = converted["point_lights"][0]
        self.assertEqual(light["camera_position"], [2.0, -1.0, 13.0])
        self.assertEqual(light["image_plane"]["depth"], 13.0)
        self.assertNotIn("u", light["image_plane"])
        self.assertFalse(
            coordinates.convert_row(row, require_intrinsics=False)[2],
        )

    def test_light_behind_the_camera_is_flagged(self) -> None:
        row = axis_aligned_row()
        row["point_lights"][0]["position"] = [1.0, -11.0, 0.0]
        converted, _, _ = coordinates.convert_row(row)
        plane = converted["point_lights"][0]["image_plane"]
        self.assertEqual(plane["depth"], -1.0)
        self.assertFalse(plane["in_front"])


class DirectoryConversionTest(unittest.TestCase):
    def test_directory_conversion_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "manifests"
            target = root / "manifests_camera"
            source.mkdir()
            (source / "train.jsonl").write_text(
                json.dumps(axis_aligned_row()) + "\n", encoding="utf-8"
            )
            stats = coordinates.convert_directory(source, target)
            self.assertEqual(stats[0][0], "train.jsonl")
            self.assertEqual(stats[0][1], 1)
            self.assertEqual(stats[0][2], 1)
            self.assertTrue(stats[0][3])
            row = json.loads((target / "train.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(row["point_lights"][0]["camera_position"], [2.0, -1.0, 13.0])
            self.assertEqual(
                row["camera_position_convention"]["axes"], "x=right,y=down,z=forward"
            )
            with self.assertRaises(ValueError):
                coordinates.convert_directory(source, target)
            self.assertEqual(len(list(target.iterdir())), 1)

    def test_parse_resolution_forms(self) -> None:
        self.assertEqual(coordinates.parse_resolution(None), None)
        self.assertEqual(coordinates.parse_resolution("960"), (960, 960))
        self.assertEqual(coordinates.parse_resolution("960x540"), (960, 540))
        self.assertEqual(coordinates.parse_resolution("960,540"), (960, 540))
        with self.assertRaises(ValueError):
            coordinates.parse_resolution("wide")


if __name__ == "__main__":
    unittest.main()
