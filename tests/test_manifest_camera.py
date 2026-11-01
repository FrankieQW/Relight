from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import coordinates  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]  # the datacompose package directory


def load_script(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_row() -> dict:
    """One composition metadata document as the Blender renderer writes it."""
    return {
        "schema_version": "1.0",
        "id": "job_1",
        "asset_uid": "uid-1",
        "asset": "objects/uid-1.glb",
        "ambient": "components/job_1/ambient.exr",
        "dark": "components/job_1/dark.exr",
        "point_lights": [
            {
                "path": "components/job_1/point_lights/light_000.exr",
                "position": [2.0, 3.0, 1.0],
                "renderer_position": [2.0, 3.0, 1.0],
                "base_energy": 100.0,
                "diffuse": 0.1,
            }
        ],
        "diffuse": [
            {"path": "components/job_1/diffuse/spread_00.exr", "level": 0.0, "size": 0.05},
            {"path": "components/job_1/diffuse/spread_05.exr", "level": 1.0, "size": 1.5},
        ],
        "in_scene_lights": [
            {
                "path": "components/job_1/in_scene_lights/fixture_000_on.exr",
                "mask": "components/job_1/in_scene_lights/fixture_000_mask.png",
                "position": [0.0, 0.0, 0.0],
                "renderer_position": [0.0, 0.0, 0.0],
                "base_energy": 50.0,
                "diffuse": 0.05,
                "fixture_source": "scene_native",
                "binding": "controlled_point_at_fixture_center",
            }
        ],
        "camera": {
            "location": [0.0, -10.0, 0.0],
            "rotation_euler": [1.5707963267948966, 0.0, 0.0],
            "focal_length": 18.0,
            "target": [0.0, 0.0, 0.0],
            "shift_x": 0.0,
            "shift_y": 0.0,
            "angle_x": 1.5707963267948966,
            "angle_y": 1.5707963267948966,
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
        "canonical": {
            "origin": [0.0, 0.0, 0.0],
            "asset_size": 1.5,
            "position_axes": "x=right,y=camera-forward,z=up",
            "units": "meter",
        },
        "composition": {"relation": "place"},
        "base_scene_id": "bistro_interior",
        "lineage": {"render_job_digest": "sha256:deadbeef"},
        "license": {"decision": "allowed", "policy_version": "operator-policy-v1"},
    }


class ManifestCameraCoordinateTest(unittest.TestCase):
    """build_manifests must ship OpenCV camera coordinates for training."""

    def test_manifests_carry_camera_position_and_image_plane(self) -> None:
        builder = load_script("tokenlight/build_manifests.py", "build_manifests_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "components" / "job_1"
            component.mkdir(parents=True)
            (component / "metadata.json").write_text(
                json.dumps(metadata_row()), encoding="utf-8"
            )
            jobs = root / "render_jobs.jsonl"
            jobs.write_text(
                json.dumps(
                    {
                        "job_id": "job_1",
                        "object_uid": "uid-1",
                        "base_scene_id": "bistro_interior",
                        "lineage": {"render_job_digest": "sha256:deadbeef"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {
                            "render_output_root": str(root),
                            "render_jobs_manifest": str(jobs),
                        },
                        "data": {
                            "split_profile": "object-held-out",
                            "require_composition_contract": True,
                        },
                        "render": {
                            "train_fraction": 0.7,
                            "validation_fraction": 0.2,
                            "test_fraction": 0.1,
                            "split_seed": 7,
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = sys.argv
            sys.argv = ["build_manifests.py", "--config", str(config_path)]
            try:
                builder.main()
            finally:
                sys.argv = previous

            row = json.loads(
                (root / "manifests" / "train.jsonl").read_text(encoding="utf-8").strip()
            )
            light = row["point_lights"][0]
            self.assertEqual(light["position"], [2.0, 3.0, 1.0])
            self.assertEqual(light["camera_position"], [2.0, -1.0, 13.0])
            self.assertEqual(light["image_plane"]["depth"], 13.0)
            self.assertAlmostEqual(light["image_plane"]["u"], 512.0 * 2.0 / 13.0 + 512.0)
            self.assertEqual(row["camera_position_convention"]["axes"], "x=right,y=down,z=forward")
            release = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
            self.assertEqual(release["camera_position"]["axes"], "x=right,y=down,z=forward")
            self.assertEqual(release["camera_position"]["rows_with_image_plane_pixels"], 1)

    def test_passes_block_passthrough_unchanged(self) -> None:
        builder = load_script("tokenlight/build_manifests.py", "build_manifests_passes")
        passes = {
            "definition": {
                "albedo": "cycles_diffuse_color",
                "irradiance": "diffuse_radiance_over_albedo",
                "normal": "camera_space_unit_opencv",
                "eps": 0.001,
            },
            "albedo": "components/job_1/passes/albedo.exr",
            "normal_camera": "components/job_1/passes/normal_camera.exr",
            "valid_mask": "components/job_1/passes/valid_mask.png",
            "material_audit": "components/job_1/passes/material_audit.json",
            "quality": {
                "base_color_linked_ratio": 0.98,
                "magenta_ratio": 0.0,
                "textured_pixel_ratio": 0.34,
            },
            "irradiance": {"ambient": "components/job_1/passes/irradiance/ambient.exr"},
        }
        row = metadata_row()
        row["passes"] = passes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "components" / "job_1"
            component.mkdir(parents=True)
            (component / "metadata.json").write_text(json.dumps(row), encoding="utf-8")
            jobs = root / "render_jobs.jsonl"
            jobs.write_text(
                json.dumps(
                    {
                        "job_id": "job_1",
                        "object_uid": "uid-1",
                        "base_scene_id": "bistro_interior",
                        "lineage": {"render_job_digest": "sha256:deadbeef"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {
                            "render_output_root": str(root),
                            "render_jobs_manifest": str(jobs),
                        },
                        "data": {
                            "split_profile": "object-held-out",
                            "require_composition_contract": True,
                        },
                        "render": {
                            "train_fraction": 0.7,
                            "validation_fraction": 0.2,
                            "test_fraction": 0.1,
                            "split_seed": 7,
                            "passes": {
                                "min_base_color_linked_ratio": 0.9,
                                "max_magenta_ratio": 0.0001,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = sys.argv
            sys.argv = ["build_manifests.py", "--config", str(config_path)]
            try:
                builder.main()
            finally:
                sys.argv = previous

            shipped = json.loads(
                (root / "manifests" / "train.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(shipped["passes"], passes)
            release = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
            self.assertTrue(release["intrinsic_passes"]["enabled"])
            self.assertEqual(release["intrinsic_passes"]["rows_with_passes"], 1)
            self.assertEqual(
                release["intrinsic_passes"]["definitions"]["albedo"], "cycles_diffuse_color"
            )

    def test_validator_rejects_missing_or_wrong_camera_position(self) -> None:
        validator = load_script(
            "tokenlight/validate_components.py", "validate_components_test"
        )
        row = metadata_row()
        converted, _, _ = coordinates.convert_row(row, require_intrinsics=False)
        validator.require_camera_position(converted, converted["point_lights"][0], "point_light")

        missing = json.loads(json.dumps(converted))
        del missing["point_lights"][0]["camera_position"]
        with self.assertRaises(ValueError):
            validator.require_camera_position(missing, missing["point_lights"][0], "point_light")

        wrong = json.loads(json.dumps(converted))
        wrong["point_lights"][0]["camera_position"] = [2.0, 3.0, 1.0]
        with self.assertRaises(ValueError):
            validator.require_camera_position(wrong, wrong["point_lights"][0], "point_light")


if __name__ == "__main__":
    unittest.main()
