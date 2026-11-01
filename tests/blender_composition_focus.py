from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import bpy


ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "blender" / "render_composition.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("focused_composition_renderer", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer: {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BlenderCompositionFocusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.renderer = load_renderer()
        cls.helpers = cls.renderer._load_helpers()
        cls.recovery = cls.renderer._load_recovery()

    def setUp(self) -> None:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    def test_generated_camera_success_and_all_candidates_rejected(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        job = self._camera_job()
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_render_mask = self.renderer._render_mask
        self.renderer._render_mask = lambda path, objects, config, **kwargs: 256
        try:
            with tempfile.TemporaryDirectory() as temporary:
                partial = Path(temporary)
                camera, _, result = self.renderer._select_camera(
                    job, [], [inserted], partial, render_config
                )
                bounds = result["inserted_ndc_bounds"]
                self.assertLessEqual(0.0, bounds["min_x"])
                self.assertLessEqual(bounds["max_x"], 1.0)
                self.assertLessEqual(0.0, bounds["min_y"])
                self.assertLessEqual(bounds["max_y"], 1.0)
                self.assertEqual(result["visible_pixels"], 256)
                self.assertEqual(result["target_visible_pixels"], 256)
                self.assertEqual(len(result["target_entity_ndc"]), 1)
                bpy.data.objects.remove(camera, do_unlink=True)

                rejected = self._camera_job()
                rejected["camera"]["ndc_x_range"] = [0.0, 0.1]
                with self.assertRaisesRegex(ValueError, "no generated camera"):
                    self.renderer._select_camera(rejected, [], [inserted], partial, render_config)
                diagnostics = partial / "diagnostics" / "camera_failures.json"
                self.assertTrue(diagnostics.is_file())
        finally:
            self.renderer._render_mask = original_render_mask

    def test_fixture_prefers_scene_native_then_uses_procedural_fallback(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        lamp = self._cube("Lamp", (0.0, 0.0, 1.2), size=0.2)
        lamp["lc_scene_entity_id"] = "scene-1:entity:lamp"
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_render_mask = self.renderer._render_mask
        self.renderer._render_mask = lambda path, objects, config, **kwargs: 256
        try:
            with tempfile.TemporaryDirectory() as temporary:
                partial = Path(temporary)
                camera, target, camera_result = self.renderer._select_camera(
                    self._camera_job(), [], [inserted], partial, render_config
                )
                runtime = {
                    "render": {"minimum_visible_pixels": 64},
                    "fixture": {"minimum_visible_pixels": 64},
                }
                native = self.renderer._select_or_create_fixture(
                    {"fixture_candidate_entity_ids": ["scene-1:entity:lamp"]},
                    runtime,
                    camera,
                    target,
                    camera_result["subject_diameter"],
                    partial,
                    self.helpers,
                    random.Random(1),
                )
                self.assertEqual(native["source"], "scene_native")
                self.assertEqual(native["entity_id"], "scene-1:entity:lamp")

                runtime["fixture"].update(
                    {
                        "fallback_candidates": 1,
                        "fallback_position_ranges": {
                            "x": [0.0, 0.0],
                            "y": [0.0, 0.0],
                            "z": [1.0, 1.0],
                        },
                    }
                )
                fallback = self.renderer._select_or_create_fixture(
                    {"fixture_candidate_entity_ids": []},
                    runtime,
                    camera,
                    target,
                    camera_result["subject_diameter"],
                    partial,
                    self.helpers,
                    random.Random(1),
                )
                self.assertEqual(fallback["source"], "procedural_fallback")
                self.assertIsNone(fallback["entity_id"])
        finally:
            self.renderer._render_mask = original_render_mask

    def test_camera_repairs_bad_initial_position_before_rejecting_job(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        job = self._camera_job()
        job["camera"]["repair_attempts_per_candidate"] = 3
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_clearance = self.renderer._camera_clearance
        original_foreground = self.renderer._near_foreground_fraction
        original_render_mask = self.renderer._render_mask
        calls = {"clearance": 0}

        def repaired_clearance(*args, **kwargs):
            calls["clearance"] += 1
            return 0.0 if calls["clearance"] == 1 else 10.0

        self.renderer._camera_clearance = repaired_clearance
        self.renderer._near_foreground_fraction = lambda *args, **kwargs: 0.0
        self.renderer._render_mask = lambda path, objects, config, **kwargs: 256
        try:
            with tempfile.TemporaryDirectory() as temporary:
                camera, _, result = self.renderer._select_camera(
                    job, [], [inserted], Path(temporary), render_config
                )
                self.assertGreater(result["repair_index"], 0)
                self.assertGreaterEqual(calls["clearance"], 2)
                bpy.data.objects.remove(camera, do_unlink=True)
        finally:
            self.renderer._camera_clearance = original_clearance
            self.renderer._near_foreground_fraction = original_foreground
            self.renderer._render_mask = original_render_mask

    def test_final_camera_rejects_oversized_fixture(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        fixture_mesh = self._cube("Fixture", (0.0, -0.5, 1.0), size=0.1)
        fixture = {"objects": [fixture_mesh]}
        job = self._camera_job()
        render_config = self.renderer._render_config({"minimum_visible_pixels": 64})
        original_clearance = self.renderer._camera_clearance
        original_foreground = self.renderer._near_foreground_fraction
        original_render_mask = self.renderer._render_mask
        self.renderer._camera_clearance = lambda *args, **kwargs: 10.0
        self.renderer._near_foreground_fraction = lambda *args, **kwargs: 0.0

        def mask_pixels(path, objects, config, **kwargs):
            return 10000 if fixture_mesh in objects else 256

        self.renderer._render_mask = mask_pixels
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "no generated camera"):
                    self.renderer._select_camera(
                        job,
                        [],
                        [inserted],
                        Path(temporary),
                        render_config,
                        fixture=fixture,
                        fixture_config={
                            "minimum_visible_pixels": 64,
                            "maximum_screen_fraction": 0.01,
                        },
                    )
        finally:
            self.renderer._camera_clearance = original_clearance
            self.renderer._near_foreground_fraction = original_foreground
            self.renderer._render_mask = original_render_mask

    def test_reference_mask_exposes_real_occlusion_ratio(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        self._cube("ForegroundBlocker", (1.5, 0.0, 0.5), size=0.75)
        camera_data = bpy.data.cameras.new("MaskCamera")
        camera = bpy.data.objects.new("MaskCamera", camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = (3.0, 0.0, 0.5)
        self.renderer._look_at(camera, (0.0, 0.0, 0.5))
        bpy.context.scene.camera = camera
        render_config = self.renderer._render_config({"resolution": 64})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            occluded = self.renderer._render_mask(
                root / "occluded.png", [inserted], render_config
            )
            reference = self.renderer._render_mask(
                root / "reference.png", [inserted], render_config, occlude=False
            )
        self.assertGreater(reference, 0)
        self.assertLess(occluded, reference)

    def test_ambient_world_is_controlled_and_hdri_selection_is_deterministic(self) -> None:
        config = self.renderer._render_config(
            {"hdri_root": None, "ambient_world_strength": 1.25}
        )
        job = {"job_id": "composition_hdri", "seed": 123}
        background, hdri = self.renderer._configure_ambient_world(config, job)
        self.assertIsNone(hdri)
        self.assertAlmostEqual(background.inputs["Strength"].default_value, 1.25)
        self.assertEqual(
            tuple(round(value, 2) for value in background.inputs["Color"].default_value),
            (0.18, 0.18, 0.18, 1.0),
        )
        self.assertTrue(background.outputs["Background"].is_linked)

        with tempfile.TemporaryDirectory() as temporary:
            hdri_root = Path(temporary)
            first = hdri_root / "first.hdr"
            second = hdri_root / "nested" / "second.EXR"
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            selected_once = self.renderer._select_ambient_hdri(hdri_root, job)
            selected_twice = self.renderer._select_ambient_hdri(hdri_root, job)
            self.assertEqual(selected_once, selected_twice)
            self.assertIn(selected_once, {first.resolve(), second.resolve()})

    def test_preflight_failure_does_not_create_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            job = self._preflight_job()
            runtime = {
                "project_root": str(root),
                "object_root": str(root / "objects"),
                "output_root": str(output_root),
                "render": {"require_verified_license": True},
            }
            with self.assertRaises(FileNotFoundError):
                self.renderer.render_job(job, runtime, self.helpers, self.recovery)
            self.assertFalse(
                (output_root / "components" / f"{job['job_id']}.partial").exists()
            )

    def test_reuse_rejects_stale_render_job_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            job = self._preflight_job()
            metadata_path = output_root / "components" / job["job_id"] / "metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                '{"id":"composition_focus","lineage":{"render_job_digest":"sha256:old"}}',
                encoding="utf-8",
            )
            runtime = {
                "project_root": str(root),
                "object_root": str(root / "objects"),
                "output_root": str(output_root),
                "overwrite": False,
            }
            with self.assertRaisesRegex(FileExistsError, "does not match"):
                self.renderer.render_job(job, runtime, self.helpers, self.recovery)

    def test_bvh_rejects_aabb_false_positive(self) -> None:
        left = self._cube("SceneLeft", (-2.0, 0.0, 0.5))
        right = self._cube("SceneRight", (2.0, 0.0, 0.5))
        bpy.ops.object.select_all(action="DESELECT")
        left.select_set(True)
        right.select_set(True)
        bpy.context.view_layer.objects.active = left
        bpy.ops.object.join()
        disconnected_scene_mesh = bpy.context.object
        disconnected_scene_mesh.name = "DisconnectedSceneMesh"
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5), size=0.5)

        inserted_minimum, inserted_maximum = self.renderer._world_bounds([inserted])
        scene_minimum, scene_maximum = self.renderer._world_bounds([disconnected_scene_mesh])
        self.assertTrue(
            all(
                min(inserted_maximum[axis], scene_maximum[axis])
                > max(inserted_minimum[axis], scene_minimum[axis])
                for axis in range(3)
            )
        )
        collisions = self.renderer._find_scene_collisions(
            [inserted],
            [disconnected_scene_mesh],
            {"collision_aabb_tolerance": 0.0, "collision_bvh_epsilon": 0.0},
        )
        self.assertEqual(collisions, [])

    def test_collision_resolution_uniformly_shrinks_and_reanchors(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        root = self._parent_under_root(inserted)
        self._cube("Obstacle", (0.6, 0.0, 0.5), size=0.4)
        target = {
            "relation": "replace",
            "bottom_center_world": [0.0, 0.0, 0.0],
            "dimensions_world": [1.0, 1.0, 1.0],
        }
        result = self.renderer._resolve_inserted_placement(
            root,
            [inserted],
            [],
            target,
            {
                "collision_aabb_tolerance": 0.0,
                "collision_bvh_epsilon": 0.0,
                "collision_shrink_enabled": True,
                "collision_shrink_factor": 0.8,
                "collision_min_scale_ratio": 0.6,
                "collision_max_attempts": 4,
            },
        )
        self.assertEqual(result["collision_pairs"], [])
        self.assertEqual(result["collision_resolution"]["strategy"], "uniform_shrink")
        self.assertAlmostEqual(result["collision_resolution"]["scale_ratio"], 0.8)
        self.assertEqual(result["collision_resolution"]["attempts"], 2)
        minimum, maximum = self.renderer._world_bounds([inserted])
        self.assertAlmostEqual(minimum.z, 0.0, places=6)
        self.assertAlmostEqual((minimum.x + maximum.x) * 0.5, 0.0, places=6)
        self.assertAlmostEqual((minimum.y + maximum.y) * 0.5, 0.0, places=6)

    def test_collision_resolution_stops_at_minimum_scale(self) -> None:
        inserted = self._cube("Inserted", (0.0, 0.0, 0.5))
        root = self._parent_under_root(inserted)
        obstacle = self._cube("Obstacle", (0.0, 0.0, 0.5))
        obstacle.scale = (0.2, 2.0, 2.0)
        bpy.context.view_layer.update()
        target = {
            "relation": "replace",
            "bottom_center_world": [0.0, 0.0, 0.0],
            "dimensions_world": [1.0, 1.0, 1.0],
        }
        with self.assertRaisesRegex(ValueError, "down to ratio 0.6"):
            self.renderer._resolve_inserted_placement(
                root,
                [inserted],
                [],
                target,
                {
                    "collision_aabb_tolerance": 0.0,
                    "collision_bvh_epsilon": 0.0,
                    "collision_shrink_enabled": True,
                    "collision_shrink_factor": 0.8,
                    "collision_min_scale_ratio": 0.6,
                    "collision_max_attempts": 4,
                },
            )
        self.assertAlmostEqual(root.scale.x, 0.6)
        self.assertAlmostEqual(root.scale.y, 0.6)
        self.assertAlmostEqual(root.scale.z, 0.6)

    @staticmethod
    def _cube(name: str, location, *, size: float = 1.0):
        bpy.ops.mesh.primitive_cube_add(size=size, location=location)
        cube = bpy.context.object
        cube.name = name
        return cube

    @staticmethod
    def _parent_under_root(obj):
        root = bpy.data.objects.new(f"Root_{obj.name}", None)
        bpy.context.collection.objects.link(root)
        matrix = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = matrix
        return root

    @staticmethod
    def _camera_job() -> dict:
        return {
            "seed": 7,
            "target": {
                "relation": "replace",
                "center_world": [0.0, 0.0, 0.5],
                "bottom_center_world": [0.0, 0.0, 0.0],
            },
            "camera": {
                "strategy": "generated_target_visible",
                "focal_length": 50.0,
                "azimuth_degrees": [0.0],
                "elevation_degrees": [30.0],
                "subject_fill_range": [0.2, 0.7],
                "shift_x_range": [0.0, 0.0],
                "shift_y_range": [0.0, 0.0],
                "ndc_x_range": [0.1, 0.9],
                "ndc_y_range": [0.1, 0.9],
                "edge_margin": 0.01,
                "candidate_count": 1,
                "target_minimum_visible_pixels": 64,
                "repair_attempts_per_candidate": 1,
                "scored_candidate_limit": 1,
                "minimum_visible_ratio": 0.7,
                "minimum_clearance": 0.01,
                "clearance_subject_ratio": 0.01,
                "foreground_grid_size": 4,
                "maximum_foreground_fraction": 0.9,
                "foreground_depth_margin_ratio": 0.1,
            },
        }

    @staticmethod
    def _preflight_job() -> dict:
        return {
            "job_id": "composition_focus",
            "base_scene_blend": "missing.blend",
            "base_scene_digest": "sha256:" + "1" * 64,
            "object_asset_path": "missing.glb",
            "prepared_geometry": {"asset_digest": "sha256:" + "2" * 64},
            "license": {"decision": "allowed"},
            "lineage": {"render_job_digest": "sha256:" + "3" * 64},
        }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderCompositionFocusTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
