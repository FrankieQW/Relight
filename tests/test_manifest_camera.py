from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import coordinates  # noqa: E402


def load_script(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_row(job_id: str = "job_1", scene_id: str = "bistro_interior") -> dict:
    """One scene-only metadata document as ``blender/render_scene.py`` writes it."""
    return {
        "schema_version": "1.0",
        "id": job_id,
        "scene_id": scene_id,
        "base_scene_blend": f"data/cache/scenes/{scene_id}.blend",
        "base_scene_digest": "sha256:" + "1" * 64,
        "ambient": f"components/{job_id}/ambient.exr",
        "dark": f"components/{job_id}/dark.exr",
        "point_lights": [
            {
                "path": f"components/{job_id}/point_lights/light_000.exr",
                "position": [2.0, 3.0, 1.0],
                "renderer_position": [2.0, 3.0, 1.0],
                "base_energy": 100.0,
                "diffuse": 0.1,
            }
        ],
        "diffuse": [
            {"path": f"components/{job_id}/diffuse/spread_00.exr", "level": 0.0, "size": 0.05},
            {"path": f"components/{job_id}/diffuse/spread_05.exr", "level": 1.0, "size": 1.5},
        ],
        "in_scene_lights": [
            {
                "path": f"components/{job_id}/in_scene_lights/fixture_000_on.exr",
                "mask": f"components/{job_id}/in_scene_lights/fixture_000_mask.png",
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
        "lineage": {"scene_camera_digest": "sha256:deadbeef"},
        "license": {"decision": "allowed", "policy_version": "operator-policy-v1"},
    }


def write_stage2_config(root: Path) -> Path:
    config_path = root / "stage2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"render_output_root": str(root), "dataset_root": str(root)},
                "data": {"split_profile": "scene-held-out"},
                "render": {
                    "train_fraction": 0.34,
                    "validation_fraction": 0.33,
                    "test_fraction": 0.33,
                    "split_seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


class SceneManifestTest(unittest.TestCase):
    """build_manifests must ship OpenCV camera coordinates, split by scene."""

    def test_manifests_carry_camera_position_and_split_by_scene(self) -> None:
        builder = load_script("tokenlight/build_manifests.py", "build_manifests_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, scene in enumerate(("bistro_interior", "bistro_exterior", "bistro_wine")):
                component = root / "components" / f"job_{index}"
                component.mkdir(parents=True)
                (component / "metadata.json").write_text(
                    json.dumps(metadata_row(f"job_{index}", scene)), encoding="utf-8"
                )
            config_path = write_stage2_config(root)

            previous = sys.argv
            sys.argv = ["build_manifests.py", "--config", str(config_path)]
            try:
                builder.main()
            finally:
                sys.argv = previous

            rows = []
            for split in ("train", "validation", "test"):
                path = root / "manifests" / f"{split}.jsonl"
                if path.is_file():
                    rows.extend(
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line
                    )
            self.assertEqual(len(rows), 3)
            # A scene must never be split across manifests.
            by_scene: dict[str, set[str]] = {}
            for split in ("train", "validation", "test"):
                path = root / "manifests" / f"{split}.jsonl"
                if not path.is_file():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line:
                        by_scene.setdefault(json.loads(line)["scene_id"], set()).add(split)
            self.assertTrue(all(len(splits) == 1 for splits in by_scene.values()), by_scene)

            row = rows[0]
            light = row["point_lights"][0]
            self.assertEqual(light["position"], [2.0, 3.0, 1.0])
            self.assertEqual(light["camera_position"], [2.0, -1.0, 13.0])
            self.assertEqual(light["image_plane"]["depth"], 13.0)
            self.assertAlmostEqual(light["image_plane"]["u"], 512.0 * 2.0 / 13.0 + 512.0)
            self.assertEqual(row["camera_position_convention"]["axes"], "x=right,y=down,z=forward")

            release = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
            self.assertEqual(release["split_profile"], "scene-held-out")
            self.assertEqual(release["counts"]["train"] + release["counts"]["validation"] + release["counts"]["test"], 3)
            self.assertEqual(release["scenes"], {split: 1 for split in ("train", "validation", "test")})
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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, scene in enumerate(("bistro_interior", "bistro_exterior", "bistro_wine")):
                row = metadata_row(f"job_{index}", scene)
                if index == 0:
                    row["passes"] = passes
                component = root / "components" / f"job_{index}"
                component.mkdir(parents=True)
                (component / "metadata.json").write_text(json.dumps(row), encoding="utf-8")
            config_path = write_stage2_config(root)

            previous = sys.argv
            sys.argv = ["build_manifests.py", "--config", str(config_path)]
            try:
                builder.main()
            finally:
                sys.argv = previous

            shipped = []
            for split in ("train", "validation", "test"):
                path = root / "manifests" / f"{split}.jsonl"
                if path.is_file():
                    shipped.extend(
                        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
                    )
            with_passes = [row for row in shipped if "passes" in row]
            self.assertEqual(len(with_passes), 1)
            self.assertEqual(with_passes[0]["passes"], passes)
            release = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
            self.assertTrue(release["intrinsic_passes"]["enabled"])
            self.assertEqual(release["intrinsic_passes"]["rows_with_passes"], 1)
            self.assertEqual(
                release["intrinsic_passes"]["definitions"]["albedo"], "cycles_diffuse_color"
            )

    def test_validator_flags_missing_component_and_scene_leakage(self) -> None:
        validator = load_script("tokenlight/validate_components.py", "validate_components_test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "components" / "job_1"
            component.mkdir(parents=True)
            row = coordinates.convert_row(metadata_row(), require_intrinsics=False)[0]
            for relative in (
                row["ambient"],
                row["dark"],
                row["point_lights"][0]["path"],
                row["diffuse"][0]["path"],
                row["diffuse"][1]["path"],
                row["in_scene_lights"][0]["path"],
                row["in_scene_lights"][0]["mask"],
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"x")
            quality = {
                "min_base_color_linked_ratio": 0.9,
                "max_magenta_ratio": 1e-4,
                "require_textures": False,
            }
            counts = {
                "point_lights": 0,
                "diffuse": 0,
                "in_scene_lights": 0,
                "scene_native_fixtures": 0,
                "procedural_fallback_fixtures": 0,
                "pass_sets": 0,
            }
            self.assertEqual(validator.validate_row(row, root, quality, counts), 7)

            with self.assertRaises(ValueError):
                validator.validate_row({**row, "scene_id": ""}, root, quality, counts)

            broken = json.loads(json.dumps(row))
            (root / broken["dark"]).unlink()
            with self.assertRaises(ValueError):
                validator.validate_row(broken, root, quality, counts)

            leaked = json.loads(json.dumps(row))
            leaked["scene_id"] = ""
            with self.assertRaises(ValueError):
                validator.validate_row(leaked, root, quality, counts)


if __name__ == "__main__":
    unittest.main()
