from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]  # the datacompose package directory
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VALIDATOR_PATH = ROOT / "tokenlight" / "validate_components.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("tokenlight_validate_components", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompositionValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = load_validator()

    def test_valid_generated_camera_and_scene_native_fixture_pass(self) -> None:
        self.validator.validate_composition_contract(valid_metadata())

    def test_missing_camera_lineage_or_fixture_contract_fails(self) -> None:
        mutations = {
            "camera": lambda row: row.pop("camera"),
            "lineage": lambda row: row.pop("lineage"),
            "fixture": lambda row: row["in_scene_lights"][0].pop("fixture_source"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                row = valid_metadata()
                mutate(row)
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    self.validator.validate_composition_contract(row)

    def test_camera_ndc_visible_pixels_and_fixture_source_regressions_fail(self) -> None:
        mutations = {
            "cropped_ndc": lambda row: row["camera"]["inserted_ndc_bounds"].update(
                {"max_x": 1.1}
            ),
            "target_pixels": lambda row: row["camera"].update({"target_visible_pixels": 0}),
            "fixture_source": lambda row: row["in_scene_lights"][0].update(
                {"fixture_source": "unknown"}
            ),
            "visible_ratio": lambda row: row["camera"].update(
                {"inserted_visible_ratio": 0.1}
            ),
            "foreground_fraction": lambda row: row["camera"].update(
                {"near_foreground_fraction": 1.1}
            ),
            "fixture_screen_fraction": lambda row: row["in_scene_lights"][0].update(
                {"screen_fraction": 0.0}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                row = valid_metadata()
                mutate(row)
                with self.assertRaises(ValueError):
                    self.validator.validate_composition_contract(row)

    def test_procedural_fixture_cannot_claim_scene_entity(self) -> None:
        row = valid_metadata()
        fixture = row["in_scene_lights"][0]
        fixture["fixture_source"] = "procedural_fallback"
        with self.assertRaisesRegex(ValueError, "不得声明 scene entity"):
            self.validator.validate_composition_contract(row)

    def test_collision_resolution_must_match_final_scale(self) -> None:
        mutations = {
            "remaining_collision": lambda row: row["composition"].update(
                {"collision_pairs": [["inserted", "obstacle"]]}
            ),
            "invalid_ratio": lambda row: row["composition"]["collision_resolution"].update(
                {"scale_ratio": 1.1}
            ),
            "scale_mismatch": lambda row: row["composition"].update(
                {"asset_scale": [0.7, 0.7, 0.7]}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                row = valid_metadata()
                mutate(row)
                with self.assertRaises(ValueError):
                    self.validator.validate_composition_contract(row)


def valid_metadata() -> dict:
    digest = "sha256:" + "a" * 64
    return {
        "schema_version": "1.0",
        "base_scene_id": "scene-1",
        "base_scene_fingerprint": digest,
        "camera": {
            "location": [2.0, 0.0, 1.0],
            "rotation_euler": [1.0, 0.0, 1.5],
            "target": [0.0, 0.0, 0.5],
            "coordinate_space": "blender_world_meter",
            "strategy": "generated_target_visible",
            "candidate_index": 0,
            "coarse_candidate_index": 0,
            "repair_index": 1,
            "target_visible_pixels": 256,
            "inserted_reference_pixels": 640,
            "inserted_visible_ratio": 0.8,
            "clearance": 0.25,
            "near_foreground_fraction": 0.1,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "inserted_ndc_bounds": {
                "min_x": 0.3,
                "max_x": 0.7,
                "min_y": 0.25,
                "max_y": 0.75,
                "width": 0.4,
                "height": 0.5,
                "center_x": 0.5,
                "center_y": 0.5,
                "min_depth": 1.0,
            },
            "target_entity_ndc": [[0.5, 0.5, 1.0]],
        },
        "canonical": {
            "origin": [0.0, 0.0, 0.5],
            "asset_size": 1.0,
            "position_axes": "x=right,y=camera-forward,z=up",
        },
        "composition": {
            "relation": "replace",
            "target_entity_id": "scene-1:entity:target",
            "object_transform_world": [1.0, 0.0, 0.0, 0.0] * 4,
            "inserted_visible_pixels": 512,
            "initial_asset_scale": [1.0, 1.0, 1.0],
            "asset_scale": [0.81, 0.81, 0.81],
            "collision_pairs": [],
            "collision_resolution": {
                "strategy": "uniform_shrink",
                "scale_ratio": 0.81,
                "attempts": 3,
                "initial_collision_count": 1,
                "initial_collision_pairs_preview": [["inserted", "obstacle"]],
                "footprint_adjusted": False,
            },
        },
        "lighting_profile": "default",
        "lineage": {
            "annotation_digest": digest,
            "object_digest": digest,
            "scene_digest": digest,
            "prepared_geometry_digest": digest,
            "base_scene_source_digest": digest,
            "render_contract_digest": digest,
            "render_job_digest": digest,
            "config_digest": digest,
            "generator_version": "focused-test",
        },
        "license": {"decision": "allowed", "policy_version": "test-v1"},
        "in_scene_lights": [
            {
                "path": "in_scene_lights/fixture_000_on.npy",
                "mask": "in_scene_lights/fixture_000_mask.png",
                "position": [0.1, -0.2, 0.8],
                "renderer_position": [0.1, 0.2, 1.3],
                "fixture_source": "scene_native",
                "fixture_entity_id": "scene-1:entity:lamp",
                "visible_pixels": 128,
                "screen_fraction": 0.01,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
