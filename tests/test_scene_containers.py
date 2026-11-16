"""Tests for the hollow-container free-space model.

The bug this guards against: an imported interior often arrives as *one* mesh
whose AABB spans the whole room (``paris_building_01_interior``, 877 m3).  Under
a solid-box model every eye-height point is "inside geometry" and a ray cast
from inside never hits the shell, so the planner produced nothing at all for a
real Bistro fragment (4400 attempts, 0 accepted).  ``container_v1`` marks such a
box hollow; ``aabb_legacy`` must still reproduce the old behaviour exactly.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose.scene_cameras import (  # noqa: E402
    CameraSpec,
    ContainerSpec,
    FreeSpace,
    _sample_scene_cameras,
    detect_containers,
    evaluate_pose,
    sample_camera_poses,
)
from datacompose.scene_targets import (  # noqa: E402
    TargetSpec,
    framability_score,
    select_targets,
)

SCRATCH = PACKAGE_DIR / ".test-tmp"


def tearDownModule() -> None:
    """Leave no scratch files behind after a run."""
    shutil.rmtree(SCRATCH, ignore_errors=True)


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
        "replaceable": category not in {"wall", "floor", "ceiling", "street"},
        "support_surface": support,
        "obb_world": {},
    }


def shell_scene(scene_id: str = "shell") -> dict:
    """A room that is *one mesh* (like an imported interior) plus a street.

    There is no floor entity: the floor is part of the shell mesh, which is why
    the old floor policy silently fell through to the free height band.
    """
    return {
        "scene_id": scene_id,
        "source_fbx": f"{scene_id}.fbx",
        "source_digest": "sha256:" + "0" * 64,
        "normalized_blend": f"data/cache/scenes/{scene_id}.blend",
        "units": "meter",
        "up_axis": "+Z",
        "nodes": [
            _box("shell", (-6.0, -5.0, 0.0), (6.0, 5.0, 4.2)),
            _box("street", (-100.0, -8.0, -0.05), (100.0, -5.0, 0.05)),
            _box("table_top", (-0.8, -0.6, 0.72), (0.8, 0.6, 0.78)),
            _box("table_leg", (-0.8, -0.6, 0.0), (0.8, 0.6, 0.72)),
            _box("cup", (-0.06, 0.05, 0.78), (0.06, 0.17, 0.90)),
        ],
        "entities": [
            _entity("e:shell", ["shell"], "paris_building_01_interior", "building"),
            _entity("e:street", ["street"], "street_segment_12", "street", support=True),
            _entity("e:cup", ["cup"], "CoffeeCup_03", "cup"),
        ],
    }


def _arrays(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    mins = np.array([node["aabb_world"]["min"] for node in scene["nodes"]], dtype=float)
    maxs = np.array([node["aabb_world"]["max"] for node in scene["nodes"]], dtype=float)
    return mins, maxs


class ContainerDetectionTest(unittest.TestCase):
    def test_room_shell_is_detected_and_furniture_is_not(self) -> None:
        mins, maxs = _arrays(shell_scene())
        flags = detect_containers(mins, maxs)
        names = ["shell", "street", "table_top", "table_leg", "cup"]
        detected = [name for name, flag in zip(names, flags) if flag]
        self.assertEqual(detected, ["shell"])
        # A long thin street is large but contains nothing of interest.
        self.assertFalse(flags[names.index("street")])

    def test_empty_crate_of_the_same_size_is_not_a_container(self) -> None:
        # Same 12x10x4.2 volume, but nothing inside it -> solid (a big crate).
        mins = np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]])
        maxs = np.array([[12.0, 10.0, 4.2], [21.0, 21.0, 21.0]])
        flags = detect_containers(mins, maxs)
        self.assertFalse(flags.any())

    def test_legacy_model_detects_nothing(self) -> None:
        mins, maxs = _arrays(shell_scene())
        flags = detect_containers(mins, maxs, ContainerSpec(model="aabb_legacy"))
        self.assertFalse(flags.any())

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContainerSpec.from_mapping({"model": "not_a_model"})

    def test_container_spec_round_trips(self) -> None:
        spec = ContainerSpec(min_extent=3.0, clearance_margin=0.25)
        self.assertEqual(
            ContainerSpec.from_mapping(spec.as_dict()).as_dict(), spec.as_dict()
        )


class InteriorFreeSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = shell_scene()
        self.space = FreeSpace.from_scene(self.scene)

    def test_eye_height_clearance_is_the_distance_to_the_wall(self) -> None:
        # (3, 3, 2) inside the 12x10x4.2 shell: the nearest face is y=5, i.e. a
        # 2 m wall distance - not "inside geometry" (0 m).
        self.assertAlmostEqual(float(self.space.clearance([[3.0, 3.0, 2.0]])[0]), 2.0)
        # Outside the shell the euclidean gap is reported as before.
        self.assertAlmostEqual(
            float(self.space.clearance([[10.0, 0.0, 1.6]])[0]), 4.0, places=6
        )

    def test_ray_from_inside_hits_the_opposite_wall(self) -> None:
        directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        depth, index = self.space.raycast([0.0, 0.0, 1.6], directions, max_distance=45.0)
        self.assertTrue((depth >= 0.0).all())
        self.assertEqual([self.space.node_ids[i] for i in index], ["shell"] * 3)
        # +X: shell face at x=6, +Y: y=5, +Z: ceiling at 4.2 minus 1.6.
        np.testing.assert_allclose(depth, [6.0, 5.0, 2.6], atol=1e-6)

    def test_solid_box_still_blocks_a_ray(self) -> None:
        # The table top is not a container (0.12 m3): a ray from beside it is
        # stopped by it instead of passing through to the far wall.
        depth, index = self.space.raycast(
            [0.0, -2.0, 0.75], np.array([[0.0, 1.0, 0.0]]), max_distance=45.0
        )
        self.assertEqual(self.space.node_ids[int(index[0])], "table_top")
        self.assertAlmostEqual(float(depth[0]), 1.4, places=6)

    def test_axis_parallel_ray_does_not_fake_hit_a_neighbouring_box(self) -> None:
        """Regression: ``min(inf, -inf)`` used to turn an empty interval into a hit.

        A ray exactly parallel to an axis and outside that box's slab can never
        enter it, but the slab test reported a hit anyway. Only pitch=0 rows of
        the ray grid hit this, so it went unnoticed.
        """
        mins = np.array([[1.0, -1.0, -1.0]])
        maxs = np.array([[2.0, 1.0, 1.0]])
        space = FreeSpace(mins, maxs, ["lonely"])
        depth, index = space.raycast(
            [0.0, 0.0, 0.0], np.array([[0.0, 1.0, 0.0]]), max_distance=45.0
        )
        self.assertEqual(int(index[0]), -1)
        self.assertLess(float(depth[0]), 0.0)

    def test_container_queries(self) -> None:
        self.assertEqual(self.space.primary_container(), 0)
        self.assertEqual(self.space.container_at([0.0, 0.0, 1.6]), 0)
        self.assertEqual(self.space.container_at([20.0, 0.0, 1.6]), -1)
        self.assertEqual(self.space.container_node_ids, ["shell"])

    def test_outside_room_only_fires_above_or_below_the_shell(self) -> None:
        # Above the roof but still over the footprint: the "looking at sky" case.
        self.assertTrue(self.space.outside_room([1.0, 1.0, 8.9]))
        self.assertTrue(self.space.outside_room([1.0, 1.0, -1.0]))
        self.assertFalse(self.space.outside_room([1.0, 1.0, 1.6]))
        # Horizontally outside: left alone, so exterior parts of mixed scenes work.
        self.assertFalse(self.space.outside_room([30.0, 1.0, 8.9]))

    def test_room_shell_beyond_walls_is_still_rejected(self) -> None:
        verdict = evaluate_pose(self.space, [1.0, 1.0, 8.9], 0.0, 0.0, CameraSpec())
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "camera_outside_room")

    def test_interior_pose_is_accepted_under_container_v1(self) -> None:
        verdict = evaluate_pose(self.space, [2.0, 2.0, 1.6], 135.0, -5.0, CameraSpec())
        self.assertTrue(verdict["ok"], str(verdict["metrics"]))
        self.assertIsNone(verdict["reason"])
        self.assertEqual(verdict["metrics"]["room_node_id"], "shell")
        # The shell is the room, not "one object dominating the frame".
        self.assertEqual(verdict["metrics"]["dominant_node_id"], None)

    def test_legacy_model_reproduces_the_old_failure(self) -> None:
        legacy = FreeSpace.from_scene(self.scene, ContainerSpec(model="aabb_legacy"))
        self.assertEqual(legacy.container_node_ids, [])
        verdict = evaluate_pose(legacy, [1.0, 1.0, 1.6], 0.0, 0.0, CameraSpec())
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "camera_inside_geometry")
        self.assertEqual(float(verdict["metrics"]["camera_clearance"]), 0.0)


class InteriorSamplingTest(unittest.TestCase):
    def test_shell_only_scene_yields_poses(self) -> None:
        scene = shell_scene("shell_only")
        result = sample_camera_poses(scene, CameraSpec(), seed=4, count=3)
        self.assertGreaterEqual(len(result["accepted"]), 1)
        self.assertEqual(result["stats"]["rooms"], ["shell"])
        self.assertEqual(result["stats"]["floor_source"], "room_shell")
        self.assertAlmostEqual(result["stats"]["floor_z"], 0.0)
        # Eye height above the *shell* floor, never above the roof.
        for row in result["accepted"]:
            self.assertLess(row["location"][2], 4.2)

    def test_legacy_model_accepts_nothing_inside_the_shell(self) -> None:
        scene = shell_scene("shell_legacy")
        spec = CameraSpec(container=ContainerSpec(model="aabb_legacy"))
        result = sample_camera_poses(scene, spec, seed=4, count=3)
        self.assertEqual(result["accepted"], [])
        self.assertIn("camera_inside_geometry", result["stats"]["reject_reasons"])
        self.assertEqual(result["stats"]["rooms"], [])

    def test_floor_entity_wins_over_the_shell(self) -> None:
        scene = shell_scene("shell_floor")
        scene["nodes"].append(_box("floor", (-6.0, -5.0, -0.10), (6.0, 5.0, 0.0)))
        scene["entities"].append(
            _entity("e:floor", ["floor"], "Floor", "floor", support=True)
        )
        result = sample_camera_poses(scene, CameraSpec(), seed=4, count=2)
        self.assertEqual(result["stats"]["floor_source"], "support_categories")

    def test_free_band_is_clamped_for_floorless_scenes(self) -> None:
        # A 20 m tall fragment with no floor and no container: the raw band would
        # put cameras 18 m up, the clamp keeps them at 1.0-2.5 m.
        scene = {
            "scene_id": "floorless",
            "nodes": [
                _box("hull_a", (-5.0, -5.0, 0.0), (5.0, 5.0, 20.0)),
                _box("hull_b", (6.0, 6.0, 0.0), (7.0, 7.0, 20.0)),
            ],
            "entities": [],
        }
        result = sample_camera_poses(scene, CameraSpec(), seed=4, count=4)
        self.assertEqual(result["stats"]["floor_source"], "free_band")
        for row in result["accepted"]:
            self.assertLessEqual(row["location"][2], 2.5 + 1e-6)

    def test_entity_strategy_plans_interior_targets(self) -> None:
        scene = shell_scene("shell_entity")
        spec = CameraSpec(strategy="entity_conditioned_v1", mix_with_free_space=0.0)
        result = _sample_scene_cameras(scene, spec, seed=7, count=3)
        self.assertGreaterEqual(len(result["accepted"]), 1)
        for row in result["accepted"]:
            self.assertEqual(row["target_label"], "CoffeeCup_03")
        self.assertEqual(result["stats"]["rooms"], ["shell"])
        self.assertEqual(result["stats"]["targets_report"]["scene_floor_z"], 0.0)


class TargetGeometryFilterTest(unittest.TestCase):
    def _targets(self, scene: dict) -> list[str]:
        report: dict = {}
        select_targets(scene, TargetSpec(), seed=0, floor_z=0.0, report=report)
        return [row["label"] for row in report["selected"]]

    def test_thin_slabs_are_not_subjects(self) -> None:
        scene = shell_scene("slabs")
        # A 1 cm doormat and a 3 m wide sheet of glass lying flat.
        scene["nodes"].append(_box("doormat", (1.0, 1.0, 0.0), (1.8, 1.4, 0.01)))
        scene["entities"].append(_entity("e:mat", ["doormat"], "Doormat_01", "prop"))
        self.assertEqual(self._targets(scene), ["CoffeeCup_03"])

    def test_terrain_sized_entities_are_not_subjects(self) -> None:
        scene = shell_scene("terrain")
        # 30 m x 30 m "planter" - big, thick enough, but not framable.
        scene["nodes"].append(_box("planter", (20.0, 20.0, 0.0), (50.0, 50.0, 1.0)))
        scene["entities"].append(_entity("e:planter", ["planter"], "Planter_07", "prop"))
        self.assertEqual(self._targets(scene), ["CoffeeCup_03"])

    def test_entity_spanning_the_scene_is_rejected(self) -> None:
        scene = shell_scene("spans")
        # Thick enough and short enough, but it spans the whole fragment.
        scene["nodes"].append(_box("beam", (-6.0, -5.0, 2.0), (6.0, 5.0, 3.0)))
        scene["entities"].append(_entity("e:beam", ["beam"], "Beam_01", "prop"))
        self.assertEqual(self._targets(scene), ["CoffeeCup_03"])

    def test_framability_ranks_a_prop_above_a_rug(self) -> None:
        spec = TargetSpec()
        rug = framability_score([1.6, 1.2, 0.02], 0.01, 0.0, True, spec)
        prop = framability_score([0.4, 0.3, 0.3], 0.55, 0.0, False, spec)
        street = framability_score([200.0, 3.0, 0.1], 0.05, 0.0, True, spec)
        self.assertGreater(prop, rug)
        self.assertGreater(prop, street)

    def test_report_explains_why_a_scene_has_no_subject(self) -> None:
        scene = shell_scene("no_subject")
        scene["nodes"] = [node for node in scene["nodes"] if node["node_id"] != "cup"]
        scene["entities"] = [
            entity for entity in scene["entities"] if entity["entity_id"] != "e:cup"
        ]
        report: dict = {}
        self.assertEqual(select_targets(scene, TargetSpec(), seed=0, report=report), [])
        self.assertGreater(report["rejected"].get("ignored_category", 0), 0)

    def test_ignore_categories_covers_streets_and_terrain(self) -> None:
        spec = TargetSpec()
        for category in ("street", "pavement", "plaza", "terrain", "grass", "water"):
            self.assertIn(category, spec.ignore_categories)


class DefaultConfigTest(unittest.TestCase):
    """``configs/default.yaml`` must stay in sync with the dataclass fields."""

    def setUp(self) -> None:
        from datacompose.config import load_config

        self.config = load_config(
            PACKAGE_DIR / "configs" / "default.yaml"
        )
        self.camera = self.config.section("m4").get("camera", {})

    def test_container_defaults_are_documented(self) -> None:
        container = ContainerSpec.from_mapping(self.camera.get("container"))
        self.assertEqual(container.model, "container_v1")
        self.assertAlmostEqual(container.min_extent, 2.0)
        self.assertAlmostEqual(container.clearance_margin, 0.10)

    def test_every_configured_container_key_exists(self) -> None:
        from dataclasses import fields

        known = {field.name for field in fields(ContainerSpec)}
        for key in (self.camera.get("container") or {}):
            self.assertIn(key, known)

    def test_every_configured_target_key_exists(self) -> None:
        from dataclasses import fields

        known = {field.name for field in fields(TargetSpec)}
        for key in self.camera.get("target") or {}:
            self.assertIn(key, known)

    def test_every_configured_camera_key_exists(self) -> None:
        from dataclasses import fields

        known = {field.name for field in fields(CameraSpec)}
        for key in self.camera:
            self.assertIn(key, known)

    def test_thresholds_have_defaults_in_the_config(self) -> None:
        target = self.camera["target"]
        for key in (
            "min_thickness",
            "flat_ratio",
            "max_dimension",
            "max_scene_diagonal_fraction",
            "sweet_dimension",
            "support_surface_penalty",
            "prefer_above_floor",
            "min_radius_fraction",
        ):
            self.assertIn(key, target)
        self.assertIn("free_height_clamp", self.camera)

    def test_every_configured_free_space_key_exists(self) -> None:
        from dataclasses import fields

        from datacompose.scene_cameras import FreeSpaceSpec

        known = {field.name for field in fields(FreeSpaceSpec)}
        block = self.camera.get("free_space") or {}
        self.assertTrue(block, "m4.camera.free_space must be documented")
        for key in block:
            self.assertIn(key, known)
        # Every dataclass field must be settable from the config, otherwise the
        # default silently wins over what the file says.
        self.assertEqual(known - set(block), set())

    def test_every_configured_probe_key_exists(self) -> None:
        from dataclasses import fields

        from datacompose.probes import ProbeSpec

        known = {field.name for field in fields(ProbeSpec)}
        block = self.config.section("m4").get("probe") or {}
        self.assertTrue(block, "m4.probe must be documented")
        for key in block:
            self.assertIn(key, known)
        # ``require_all``/``resume`` are booleans the launcher can override, but
        # every value still has to have a default in the file.
        self.assertEqual(known - set(block), set())

    def test_probe_defaults_are_neutral(self) -> None:
        """The new stage must not change a dataset unless someone opts in."""
        from datacompose.probes import ProbeSpec
        from datacompose.scene_cameras import CameraSpec, FreeSpaceSpec
        from datacompose.scene_targets import TargetSpec

        spec = FreeSpaceSpec.from_mapping(self.camera.get("free_space"))
        self.assertEqual(spec.source, "auto")
        self.assertEqual(spec, FreeSpaceSpec())
        self.assertEqual(
            float(self.camera.get("coverage_weight", -1.0)), float(CameraSpec().coverage_weight)
        )
        self.assertEqual(
            float(self.camera.get("min_coverage", -1.0)), float(CameraSpec().min_coverage)
        )
        self.assertEqual(
            float(
                self.camera["target"].get(
                    "azimuth_novelty_weight", -1.0
                )
            ),
            float(TargetSpec().azimuth_novelty_weight),
        )
        self.assertEqual(
            ProbeSpec.from_mapping(self.config.section("m4").get("probe")), ProbeSpec()
        )


class JobReportTest(unittest.TestCase):
    """The renderer needs the container node ids to skip shell dominance."""

    def test_free_space_report_names_the_container(self) -> None:
        root = SCRATCH / "container_job"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        blend = root / "shell.blend"
        blend.write_bytes(b"fake-blend")
        scene = shell_scene("shell_job")
        scene["normalized_blend"] = str(blend)
        from datacompose.scene_cameras import build_scene_camera_jobs
        from datacompose import schemas

        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(strategy="entity_conditioned_v1", mix_with_free_space=0.5),
            seed=21,
            cameras_per_scene=2,
        )
        schemas.SceneCameraJobDocument.model_validate(document)
        self.assertTrue(document["jobs"])
        for job in document["jobs"]:
            report = job["free_space_report"]
            self.assertEqual(report["container_node_ids"], ["shell"])
            self.assertEqual(report["room_node_id"], "shell")
        self.assertEqual(
            document["camera_spec"]["container"]["model"], "container_v1"
        )


if __name__ == "__main__":
    unittest.main()
