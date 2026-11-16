"""Tests for entity-conditioned camera planning.

The bug this guards against: the free-space planner rewarded "two walls fill the
frame" (no single node dominates, large depth spread), so a whole batch could
come back as wall-corner photos with no object in them.  Entity-conditioned
sampling picks a named target and requires it to be genuinely visible.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import sys
import unittest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import schemas  # noqa: E402
from datacompose.scene_cameras import (  # noqa: E402
    CameraSpec,
    FreeSpace,
    _sample_scene_cameras,
    build_scene_camera_jobs,
    camera_count_split,
    evaluate_target_pose,
    sample_targeted_poses,
)
from datacompose.scene_targets import (  # noqa: E402
    TargetSpec,
    occluder_exclusions,
    select_targets,
)
from datacompose.tests.test_scene_cameras import room_scene  # noqa: E402


SCRATCH = PACKAGE_DIR / ".test-tmp"


def tearDownModule() -> None:
    """Leave no scratch files behind after a run."""
    shutil.rmtree(SCRATCH, ignore_errors=True)


def scratch_dir(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _box(node_id: str, low, high) -> dict:
    return {
        "node_id": node_id,
        "raw_import_name": node_id,
        "source_fbx_element_id": None,
        "object_type": "MESH",
        "transform_world": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "aabb_world": {"min": list(low), "max": list(high)},
        "obb_world": {},
        "materials": [],
        "triangle_count": 12,
        "vertex_count": 8,
    }


def _entity(entity_id: str, node_ids, label: str, category: str, *, support=False) -> dict:
    return {
        "entity_id": entity_id,
        "node_ids": list(node_ids),
        "raw_label": label,
        "category": category,
        "category_confidence": 1.0,
        "grouping_confidence": 1.0,
        "grouping_method": "single_node",
        "replaceable": category not in {"wall", "floor", "ceiling"},
        "support_surface": support,
        "obb_world": {},
    }


ARCHITECTURE_ENTITIES = [
    _entity("e:floor", ["floor"], "Floor", "floor", support=True),
    _entity("e:ceiling", ["ceiling"], "Ceiling", "ceiling"),
    _entity("e:wall_x_neg", ["wall_x_neg"], "Wall", "wall"),
    _entity("e:wall_x_pos", ["wall_x_pos"], "Wall", "wall"),
    _entity("e:wall_y_neg", ["wall_y_neg"], "Wall", "wall"),
    _entity("e:wall_y_pos", ["wall_y_pos"], "Wall", "wall"),
]


def corner_scene(scene_id: str = "corner") -> dict:
    """Room with nothing in it: the wall-corner failure mode."""
    scene = room_scene(scene_id)
    scene["nodes"] = [node for node in scene["nodes"] if node["node_id"] != "table"]
    scene["entities"] = list(ARCHITECTURE_ENTITIES)
    return scene


def cup_scene(scene_id: str = "cup") -> dict:
    """Room + table + a 12 cm cup standing on it.

    The table deliberately has no entity, so it is pure occlusion geometry and
    the only framable target is the cup.
    """
    scene = room_scene(scene_id)
    scene["nodes"].append(_box("cup", (-0.06, -0.06, 0.75), (0.06, 0.06, 0.87)))
    scene["entities"] = list(ARCHITECTURE_ENTITIES) + [
        _entity("e:cup", ["cup"], "CoffeeCup_01", "cup")
    ]
    return scene


def ceiling_lid_scene(scene_id: str = "lid") -> dict:
    """Cup with a solid lid directly above it: nothing can see it."""
    scene = cup_scene(scene_id)
    scene["nodes"].append(_box("lid", (-1.0, -1.0, 1.20), (1.0, 1.0, 1.30)))
    return scene


def target_spec(**overrides) -> TargetSpec:
    """Deterministic spec for tests: no azimuth jitter, fixed eye direction."""
    base = TargetSpec(azimuth_jitter_degrees=0.0, attempts_per_target=80)
    return replace(base, **overrides) if overrides else base


class SelectionTest(unittest.TestCase):
    def test_corner_scene_has_no_framable_target(self) -> None:
        self.assertEqual(select_targets(corner_scene(), target_spec()), [])

    def test_cup_is_selected_over_architecture(self) -> None:
        targets = select_targets(cup_scene(), target_spec())
        self.assertEqual([target.entity_id for target in targets], ["e:cup"])
        self.assertAlmostEqual(targets[0].center[2], 0.81, places=3)
        self.assertGreater(len(targets[0].samples), 0)

    def test_tiny_entities_are_skipped(self) -> None:
        scene = cup_scene()
        scene["nodes"] = [
            node
            for node in scene["nodes"]
            if node["node_id"] != "cup"
        ] + [_box("cup", (-0.02, -0.02, 0.75), (0.02, 0.02, 0.79))]
        self.assertEqual(select_targets(scene, target_spec()), [])

    def test_transparent_and_thin_nodes_are_not_occluders(self) -> None:
        scene = cup_scene()
        scene["nodes"].append(_box("pane", (1.0, -1.0, 0.0), (1.2, 1.0, 2.0)))
        scene["entities"].append(_entity("e:pane", ["pane"], "Window", "window"))
        # A 8 mm sheet (a sign, a railing) must not count as a wall either.
        scene["nodes"].append(_box("sign", (0.0, 1.5, 1.0), (0.4, 1.508, 1.3)))
        scene["entities"].append(_entity("e:sign", ["sign"], "Sign", "prop"))
        excluded = occluder_exclusions(scene, target_spec())
        self.assertIn("pane", excluded)  # transparent category
        self.assertIn("sign", excluded)  # thinner than thin_occluder_thickness
        self.assertNotIn("cup", excluded)
        self.assertNotIn("wall_x_neg", excluded)  # 5 cm walls stay solid


class TargetedSamplingTest(unittest.TestCase):
    def test_corner_scene_yields_no_camera(self) -> None:
        result = sample_targeted_poses(corner_scene(), CameraSpec(), seed=3, count=2)
        self.assertEqual(result["accepted"], [])
        self.assertIn("no_valid_target", [row["reason"] for row in result["rejects"]])

    def test_cup_camera_is_visible_and_framed(self) -> None:
        spec = CameraSpec(target=target_spec())
        result = sample_targeted_poses(cup_scene(), spec, seed=3, count=2)
        self.assertGreaterEqual(len(result["accepted"]), 1)
        for metrics in result["accepted"]:
            self.assertEqual(metrics["target_entity_id"], "e:cup")
            self.assertGreaterEqual(
                metrics["target_visible_ratio"],
                spec.target.min_target_visible_ratio,
            )
            self.assertGreaterEqual(
                metrics["target_screen_ratio"],
                spec.target.min_target_screen_ratio,
            )
            self.assertLessEqual(
                metrics["target_screen_ratio"],
                spec.target.max_target_screen_ratio,
            )
            # The anchor is the subject centre, not a point floating in the room.
            self.assertAlmostEqual(metrics["anchor"][2], 0.81, places=3)

    def test_hidden_cup_is_rejected(self) -> None:
        scene = ceiling_lid_scene()
        spec = CameraSpec(target=target_spec())
        target = select_targets(scene, spec.target)[0]
        space = FreeSpace.from_scene(scene)
        verdict = evaluate_target_pose(
            space, target, [0.0, 0.0, 2.6], -90.0, 30.0, spec
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "target_occluded")
        self.assertLess(verdict["metrics"]["target_visible_ratio"], 0.4)

    def test_contact_tolerance_never_loses_visibility(self) -> None:
        scene = cup_scene()
        spec = CameraSpec(target=target_spec())
        target = select_targets(scene, spec.target)[0]
        space = FreeSpace.from_scene(scene)
        strict = CameraSpec(
            target=target_spec(contact_tolerance=0.0)
        )
        tolerant = evaluate_target_pose(
            space, target, [0.45, 0.0, 1.15], -90.0, 32.0, spec
        )
        exact = evaluate_target_pose(
            space, target, [0.45, 0.0, 1.15], -90.0, 32.0, strict
        )
        self.assertGreaterEqual(
            tolerant["metrics"]["target_visible_ratio"],
            exact["metrics"]["target_visible_ratio"],
        )

    def test_small_target_still_yields_several_cameras(self) -> None:
        """Orbit spacing must be angular.

        A 12 cm cup is orbited at 0.6 m, so two poses are at most 1.2 m apart:
        a 1.5 m euclidean ``min_camera_spacing`` used to allow exactly one
        camera per target, which is why a 32-camera scene collapsed to a couple
        of jobs.
        """
        spec = CameraSpec(target=target_spec(), min_camera_spacing=1.50)
        result = sample_targeted_poses(cup_scene(), spec, seed=3, count=6)
        self.assertEqual(len(result["accepted"]), 6)
        self.assertGreater(spec.target.min_azimuth_separation_degrees, 0.0)

    def test_azimuth_separation_can_be_disabled(self) -> None:
        spec = CameraSpec(target=target_spec(min_azimuth_separation_degrees=0.0))
        result = sample_targeted_poses(cup_scene(), spec, seed=3, count=4)
        self.assertNotIn("camera_azimuth_too_close", result["stats"]["reject_reasons"])

    def test_pose_behind_a_wall_is_rejected(self) -> None:
        scene = cup_scene()
        spec = CameraSpec(target=target_spec())
        target = select_targets(scene, spec.target)[0]
        space = FreeSpace.from_scene(scene)
        # Outside the room, looking in through the -X wall.
        verdict = evaluate_target_pose(
            space, target, [-4.0, 0.0, 0.81], 0.0, 0.0, spec
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "target_occluded")


class MixingTest(unittest.TestCase):
    def test_split_conserves_cameras(self) -> None:
        self.assertEqual(camera_count_split(10, 0.3), (7, 3))
        self.assertEqual(camera_count_split(1, 0.3), (1, 0))
        for count in range(1, 25):
            targeted, free = camera_count_split(count, 0.3)
            self.assertEqual(targeted + free, count)
            self.assertGreaterEqual(targeted, 1)

    def test_entity_strategy_merges_both_halves(self) -> None:
        spec = CameraSpec(
            strategy="entity_conditioned_v1",
            mix_with_free_space=0.3,
            target=target_spec(),
        )
        result = _sample_scene_cameras(cup_scene(), spec, seed=5, count=4)
        stats = result["stats"]
        self.assertEqual(stats["targeted_requested"], 3)
        self.assertEqual(stats["free_space_requested"], 1)
        self.assertGreaterEqual(stats["targeted_accepted"], 1)
        self.assertEqual(
            stats["accepted"], len(result["accepted"])
        )

    def test_default_strategy_emits_no_target_fields(self) -> None:
        root = scratch_dir("default_strategy")
        blend = root / "corner.blend"
        blend.write_bytes(b"fake-blend")
        scene = cup_scene("cup_default")
        scene["normalized_blend"] = str(blend)
        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(),
            seed=11,
            cameras_per_scene=2,
        )
        self.assertTrue(document["jobs"])
        for job in document["jobs"]:
            self.assertEqual(job["camera"]["strategy"], "scene_free_space_v1")
            self.assertNotIn("target_entity_id", job["camera"])
        self.assertEqual(document["camera_spec"]["strategy"], "scene_free_space_v1")


class EntityJobContractTest(unittest.TestCase):
    def _jobs(self, name: str) -> dict:
        root = scratch_dir(name)
        blend = root / "cup.blend"
        blend.write_bytes(b"fake-blend")
        scene = cup_scene("cup_entity")
        scene["normalized_blend"] = str(blend)
        spec = CameraSpec(
            strategy="entity_conditioned_v1",
            mix_with_free_space=0.25,
            target=target_spec(),
        )
        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=spec,
            seed=13,
            cameras_per_scene=4,
        )
        schemas.SceneCameraJobDocument.model_validate(document)
        return document

    def test_target_metadata_is_recorded(self) -> None:
        document = self._jobs("entity_jobs")
        target_rows = [
            job for job in document["jobs"] if "target_entity_id" in job["camera"]
        ]
        self.assertTrue(target_rows)
        job = target_rows[0]
        self.assertEqual(job["camera"]["target_entity_id"], "e:cup")
        self.assertEqual(job["camera"]["target_label"], "CoffeeCup_01")
        self.assertEqual(job["canonical"]["origin"], job["camera"]["target"])
        self.assertEqual(job["canonical"]["target_entity_id"], "e:cup")
        self.assertAlmostEqual(job["canonical"]["target_radius"], 0.1039, places=3)
        self.assertIn("cup", job["canonical"]["target_node_ids"])
        self.assertIn("target", job["lineage"])
        # asset_size keeps its original meaning, so the light rig still scales.
        self.assertGreater(job["canonical"]["asset_size"], 0.0)

    def test_camera_spec_records_target_block(self) -> None:
        document = self._jobs("entity_spec")
        target_block = document["camera_spec"]["target"]
        self.assertTrue(target_block["enabled"])
        self.assertEqual(target_block["min_target_visible_ratio"], 0.40)
        self.assertEqual(
            document["camera_spec"]["strategy"], "entity_conditioned_v1"
        )

    def test_bare_corner_scene_produces_no_jobs(self) -> None:
        root = scratch_dir("corner_no_jobs")
        blend = root / "corner.blend"
        blend.write_bytes(b"fake-blend")
        scene = corner_scene("corner_only")
        scene["normalized_blend"] = str(blend)
        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(
                strategy="entity_conditioned_v1", target=target_spec()
            ),
            seed=17,
            cameras_per_scene=2,
            require_all_scenes=False,
        )
        self.assertEqual(document["jobs"], [])
        self.assertTrue(
            any(row["reason"] == "no_valid_camera_pose" for row in document["rejects"])
        )


if __name__ == "__main__":
    unittest.main()
