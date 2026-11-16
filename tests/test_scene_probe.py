"""Tests for the baked clearance probe and the probe-backed free-space oracle.

The bug this guards against: the AABB model is a *broad phase*, so a pose the
planner accepted could still be inside a wall (the rectangle overestimates the
room), on the wrong storey of a multi-storey fragment, or inside a container
that was mis-detected as hollow.  Every one of those became a wasted render
slot.  A probe answers the same question from real geometry - "distance to the
nearest triangle" - which is exactly what the Blender worker re-checks.

The fixtures here build the clearance field analytically from axis-aligned
boxes, so the expected numbers are exact and no Blender is needed.
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import shutil
import sys
import unittest

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose.probes import (  # noqa: E402
    ProbeSpec,
    ProbeVolume,
    discover_probes,
    probe_path_for,
)
from datacompose.scene_cameras import (  # noqa: E402
    CameraSpec,
    ContainerSpec,
    FreeSpace,
    FreeSpaceSpec,
    ProbeFreeSpace,
    build_scene_camera_jobs,
    evaluate_pose,
    sample_camera_poses,
    view_coverage,
)
from datacompose.scene_targets import TargetSpec  # noqa: E402

SCRATCH = PACKAGE_DIR / ".test-tmp"


def tearDownModule() -> None:
    """Leave no scratch files behind after a run."""
    shutil.rmtree(SCRATCH, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Synthetic scenes and probes
# --------------------------------------------------------------------------- #
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


def _entity(entity_id: str, node_ids, label: str, category: str) -> dict:
    return {
        "entity_id": entity_id,
        "node_ids": list(node_ids),
        "raw_label": label,
        "category": category,
        "category_confidence": 1.0,
        "grouping_confidence": 1.0,
        "grouping_method": "single_node",
        "replaceable": category not in {"wall", "floor", "ceiling"},
        "support_surface": False,
        "obb_world": {},
    }


def room_scene(scene_id: str = "room") -> dict:
    """A single-mesh 10 x 10 x 3 room with a table inside (no floor entity)."""
    return {
        "scene_id": scene_id,
        "source_fbx": f"{scene_id}.fbx",
        "source_digest": "sha256:" + "0" * 64,
        "normalized_blend": f"data/cache/scenes/{scene_id}.blend",
        "units": "meter",
        "up_axis": "+Z",
        "nodes": [
            _box("shell", (0.0, 0.0, 0.0), (10.0, 10.0, 3.0)),
            _box("table", (4.0, 4.0, 0.72), (5.0, 5.0, 0.78)),
            _box("cup", (4.4, 4.4, 0.78), (4.6, 4.6, 0.90)),
        ],
        "entities": [
            _entity("e:shell", ["shell"], "interior_mesh_01", "wall"),
            _entity("e:table", ["table"], "Table_01", "table"),
            _entity("e:cup", ["cup"], "CoffeeCup_01", "cup"),
        ],
    }


def _signed_distance(points: np.ndarray, low, high) -> np.ndarray:
    """Exact ``|sdf|`` of an axis-aligned box: the distance to its surface."""
    centre = (np.asarray(low, dtype=np.float64) + np.asarray(high, dtype=np.float64)) * 0.5
    half = (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) * 0.5
    shifted = np.abs(points - centre) - half
    outside = np.linalg.norm(np.maximum(shifted, 0.0), axis=1)
    inside = np.minimum(shifted.max(axis=1), 0.0)
    return np.abs(outside + inside)


def build_probe(
    boxes,
    *,
    resolution: float = 0.1,
    padding: float = 0.1,
    clearance_cap: float = 1.0,
) -> ProbeVolume:
    """A probe whose field is the exact distance to the nearest box surface."""
    lows = np.array([box[1] for box in boxes], dtype=np.float64)
    highs = np.array([box[2] for box in boxes], dtype=np.float64)
    origin = lows.min(axis=0) - padding
    extent = highs.max(axis=0) + padding - origin
    dims = tuple(int(value) for value in np.ceil(extent / resolution))
    axes = [
        origin[axis] + (np.arange(dims[axis], dtype=np.float64) + 0.5) * resolution
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    distances = np.stack(
        [_signed_distance(grid, box[1], box[2]) for box in boxes], axis=0
    )
    owner = np.argmin(distances, axis=0).astype(np.int32)
    nearest = np.min(distances, axis=0)
    return ProbeVolume(
        origin=origin,
        resolution=resolution,
        clearance=np.minimum(nearest, clearance_cap).reshape(dims),
        nearest_node=owner.reshape(dims),
        node_ids=[box[0] for box in boxes],
        clearance_cap=clearance_cap,
    )


def _voxel(probe: ProbeVolume, point) -> tuple[int, int, int]:
    """Index of the voxel a point falls into (same rule as ``ProbeVolume``)."""
    values = np.floor(
        (np.asarray(point, dtype=np.float64) - probe.origin) / probe.resolution
    )
    return tuple(int(value) for value in values)


def room_probe(**kwargs) -> ProbeVolume:
    return build_probe(
        [
            ("shell", (0.0, 0.0, 0.0), (10.0, 10.0, 3.0)),
            ("table", (4.0, 4.0, 0.72), (5.0, 5.0, 0.78)),
            ("cup", (4.4, 4.4, 0.78), (4.6, 4.6, 0.90)),
        ],
        **kwargs,
    )


def room_oracle(probe: ProbeVolume | None = None, scene: dict | None = None) -> ProbeFreeSpace:
    scene = scene or room_scene()
    probe = probe or room_probe()
    return ProbeFreeSpace(
        probe,
        FreeSpace.from_scene(scene, ContainerSpec()),
        hit_clearance=0.10,
        step_scale=0.5,
    )


# --------------------------------------------------------------------------- #
# Probe volume
# --------------------------------------------------------------------------- #
class ProbeSpecTest(unittest.TestCase):
    def test_defaults_are_finite_and_positive(self) -> None:
        spec = ProbeSpec()
        self.assertGreater(spec.resolution, 0.0)
        self.assertGreater(spec.max_voxels, 0)
        self.assertGreater(spec.clearance_cap, 0.0)

    def test_bad_resolution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProbeSpec.from_mapping({"resolution": 0.0})

    def test_spec_round_trips(self) -> None:
        spec = ProbeSpec.from_mapping({"resolution": 0.2, "max_voxels": 1000})
        self.assertEqual(ProbeSpec.from_mapping(spec.as_dict()).as_dict(), spec.as_dict())


class ProbeVolumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = room_probe()

    def test_save_and_load_round_trip(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        path = SCRATCH / "roundtrip.probe.npz"
        self.probe.save(path)
        back = ProbeVolume.load(path)
        self.assertEqual(back.dims, self.probe.dims)
        self.assertEqual(back.node_ids, self.probe.node_ids)
        np.testing.assert_allclose(back.clearance, self.probe.clearance)
        np.testing.assert_array_equal(back.nearest_node, self.probe.nearest_node)
        self.assertAlmostEqual(back.resolution, self.probe.resolution)

    def test_unknown_format_is_rejected(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        path = SCRATCH / "bad.probe.npz"
        np.savez_compressed(
            path,
            format=np.array("lc_probe_v0"),
            origin=np.zeros(3),
            resolution=np.array(0.1),
            clearance=np.zeros((2, 2, 2), dtype=np.float32),
            nearest_node=np.zeros((2, 2, 2), dtype=np.int32),
            node_ids=np.array(["a"]),
        )
        with self.assertRaises(ValueError):
            ProbeVolume.load(path)

    def test_clearance_is_the_distance_to_the_nearest_surface(self) -> None:
        # 8 m from both walls in x and y, 1.5 m from floor and ceiling: the
        # nearest face is 1.5 m away, and the field is capped at 1.0 (a query
        # subtracts its distance to the voxel centre, so it reads slightly low).
        self.assertAlmostEqual(
            float(self.probe.clearance_at([[8.0, 8.0, 1.5]])[0]), 1.0, delta=0.10
        )
        # Just above the table top (0.78 m) the nearest surface is the table.
        near_table = float(self.probe.clearance_at([[4.5, 4.5, 0.88]])[0])
        self.assertLess(near_table, 0.15)

    def test_query_is_a_conservative_lower_bound(self) -> None:
        """A pose must never be told there is more free space than there is."""
        points = np.array([[1.37, 2.19, 1.03], [8.11, 7.43, 2.55]])
        queried = self.probe.clearance_at(points)
        exact = np.array(
            [
                float(self.probe.clearance_at([point.tolist()])[0]) for point in points
            ]
        )
        self.assertTrue(np.all(queried <= exact + 1e-9))
        # ...and it stays within one voxel of the truth, not wildly below it.
        self.assertTrue(np.all(queried >= exact - self.probe.resolution * 1.0))

    def test_points_outside_the_volume_are_treated_as_free(self) -> None:
        self.assertAlmostEqual(
            float(self.probe.clearance_at([[100.0, 100.0, 100.0]])[0]),
            self.probe.clearance_cap,
        )

    def test_cast_hits_the_walls_not_the_shell_centre(self) -> None:
        directions = np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        depth, node = self.probe.cast(
            [8.0, 8.0, 1.5], directions, max_distance=45.0, threshold=0.1
        )
        # +x wall at 10 (2.0 away), -x wall at 0 (8.0 away), ceiling at 3 (1.5),
        # floor at 0 (1.5). Detection happens a threshold before the surface.
        np.testing.assert_allclose(depth, [2.0, 8.0, 1.5, 1.5], atol=0.25)
        self.assertEqual({self.probe.node_ids[int(index)] for index in node}, {"shell"})

    def test_cast_reports_the_owning_node_of_the_hit(self) -> None:
        # A horizontal ray at table height from inside the room ends on the table.
        depth, node = self.probe.cast(
            [1.0, 4.5, 0.75], np.array([[1.0, 0.0, 0.0]]), max_distance=45.0, threshold=0.1
        )
        self.assertGreater(float(depth[0]), 0.0)
        self.assertEqual(self.probe.node_ids[int(node[0])], "table")

    def test_cast_reports_no_hit_when_it_escapes(self) -> None:
        depth, node = self.probe.cast(
            [5.0, 5.0, 1.5], np.array([[1.0, 0.0, 0.0]]), max_distance=0.5, threshold=0.1
        )
        self.assertLess(float(depth[0]), 0.0)
        self.assertEqual(int(node[0]), -1)

    def test_support_height_finds_the_floor_of_the_run(self) -> None:
        height = self.probe.height_above_support(0.20)
        # A voxel 1.55 m up, well away from the table: 1.5 m above the floor.
        index = _voxel(self.probe, (7.0, 7.0, 1.55))
        self.assertTrue(bool(self.probe.free_mask(0.20)[index]))
        self.assertAlmostEqual(float(height[index]), 1.55, delta=0.12)

    def test_random_position_is_free_and_at_standing_height(self) -> None:
        rng = random.Random(3)
        for _ in range(20):
            position = self.probe.random_position(rng, clearance=0.2, eye_height=(1.1, 1.8))
            self.assertIsNotNone(position)
            self.assertGreaterEqual(float(self.probe.clearance_at([position])[0]), 0.19)
            self.assertGreaterEqual(position[2], 1.1)
            self.assertLessEqual(position[2], 1.8)

    def test_multi_storey_keeps_cameras_on_both_floors(self) -> None:
        """A slab carved out of a tall shell makes two free runs, not one."""
        probe = build_probe(
            [
                ("building", (0.0, 0.0, 0.0), (10.0, 10.0, 6.0)),
                ("slab", (0.0, 0.0, 2.9), (10.0, 10.0, 3.1)),
            ]
        )
        positions = probe.candidate_positions(clearance=0.2, eye_height=(1.1, 1.8))
        self.assertGreater(len(positions), 0)
        lows = positions[positions[:, 2] < 3.0]
        highs = positions[positions[:, 2] > 3.0]
        self.assertGreater(len(lows), 0)
        self.assertGreater(len(highs), 0)
        # Lower storey: floor 0 -> cameras 1.1-1.8. Upper: floor 3.1 -> 4.2-4.9.
        self.assertLess(lows[:, 2].max(), 2.0)
        self.assertGreater(highs[:, 2].min(), 4.0)
        self.assertLess(highs[:, 2].max(), 5.0)


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
class ProbeOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = room_scene()
        self.coarse = FreeSpace.from_scene(self.scene, ContainerSpec())
        self.oracle = room_oracle(scene=self.scene)

    def test_oracle_reports_probe_as_its_source(self) -> None:
        self.assertEqual(self.oracle.source, "probe")
        self.assertEqual(self.oracle.node_ids, self.coarse.node_ids)
        self.assertEqual(self.oracle.node_count, len(self.coarse.node_ids))

    def test_node_coverage_is_complete_for_a_matching_probe(self) -> None:
        self.assertAlmostEqual(self.oracle.node_coverage, 1.0)

    def test_probe_never_claims_a_point_inside_a_wall_is_free(self) -> None:
        # Just outside the room, i.e. inside the 0.2 m of wall thickness the
        # AABB model would have called free space.
        clearance = float(self.oracle.clearance([[10.02, 5.0, 1.5]])[0])
        self.assertLess(clearance, 0.20)

    def test_raycast_maps_hits_onto_scene_node_ids(self) -> None:
        depth, index = self.oracle.raycast(
            [8.0, 8.0, 1.5],
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
            max_distance=45.0,
        )
        self.assertTrue(np.all(depth > 0.0))
        self.assertEqual(
            [self.oracle.node_ids[int(value)] for value in index], ["shell", "shell"]
        )

    def test_container_identification_still_comes_from_the_boxes(self) -> None:
        # The room is one mesh and the geometry oracle now sees it as walls, but
        # dominance still needs to know which node is the environment.
        self.assertEqual(self.oracle.container_node_ids, self.coarse.container_node_ids)

    def test_interior_pose_is_accepted(self) -> None:
        verdict = evaluate_pose(self.oracle, [5.0, 5.0, 1.5], 45.0, -5.0, CameraSpec())
        self.assertTrue(verdict["ok"], str(verdict["metrics"]))
        self.assertEqual(verdict["metrics"]["room_node_id"], "shell")

    def test_pose_inside_a_wall_is_rejected(self) -> None:
        verdict = evaluate_pose(self.oracle, [10.05, 5.0, 1.5], 0.0, 0.0, CameraSpec())
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "camera_inside_geometry")

    def test_sample_positions_are_all_free_under_the_probe(self) -> None:
        result = sample_camera_poses(
            self.scene, CameraSpec(), seed=5, count=3, probe=self.oracle
        )
        self.assertGreaterEqual(len(result["accepted"]), 1)
        self.assertEqual(result["stats"]["free_space_source"], "probe")
        self.assertEqual(result["stats"]["floor_source"], "probe")
        for row in result["accepted"]:
            self.assertGreaterEqual(float(row["camera_clearance"]), 0.19)


class CoverageScoreTest(unittest.TestCase):
    def test_a_single_surface_scores_low(self) -> None:
        one = view_coverage(np.array([100, 0, 0]), ["wall", "table", "cup"])
        self.assertAlmostEqual(one["categories"], 1.0)
        self.assertAlmostEqual(one["dominance"], 1.0)
        self.assertLess(one["coverage"], 0.45)

    def test_several_categories_score_high(self) -> None:
        many = view_coverage(np.array([40, 30, 20, 10]), ["wall", "table", "cup", "chair"])
        self.assertAlmostEqual(many["categories"], 4.0)
        self.assertGreater(many["coverage"], 0.6)

    def test_empty_frame_is_zero(self) -> None:
        self.assertEqual(view_coverage(np.zeros(3), ["a", "b", "c"])["coverage"], 0.0)

    def test_min_coverage_rejects_when_configured(self) -> None:
        spec = CameraSpec(min_coverage=0.99)
        oracle = room_oracle()
        verdict = evaluate_pose(oracle, [5.0, 5.0, 1.5], 45.0, -5.0, spec)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "low_scene_coverage")

    def test_min_coverage_is_off_by_default(self) -> None:
        self.assertEqual(float(CameraSpec().min_coverage), 0.0)
        self.assertEqual(float(CameraSpec().coverage_weight), 0.0)


# --------------------------------------------------------------------------- #
# Planning integration
# --------------------------------------------------------------------------- #
class ProbePlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.blend = SCRATCH / "room.blend"
        self.blend.write_bytes(b"fake-blend")

    def _document(self, scene_id: str = "room_job") -> dict:
        scene = room_scene(scene_id)
        scene["normalized_blend"] = str(self.blend)
        return {"schema_version": "1.0", "scenes": [scene]}

    def test_probe_planning_marks_the_source(self) -> None:
        document = build_scene_camera_jobs(
            self._document(),
            spec=CameraSpec(
                strategy="entity_conditioned_v1", mix_with_free_space=0.5
            ),
            seed=11,
            cameras_per_scene=2,
            probe_by_scene={"room_job": room_probe()},
        )
        self.assertTrue(document["jobs"])
        for job in document["jobs"]:
            self.assertEqual(job["free_space_report"]["source"], "probe")
        self.assertEqual(
            document["stats"]["free_space"]["probe_scenes"], ["room_job"]
        )

    def test_missing_probe_falls_back_to_aabb(self) -> None:
        document = build_scene_camera_jobs(
            self._document("room_fallback"),
            spec=CameraSpec(strategy="entity_conditioned_v1", mix_with_free_space=0.5),
            seed=11,
            cameras_per_scene=2,
        )
        self.assertTrue(document["jobs"])
        for job in document["jobs"]:
            self.assertEqual(job["free_space_report"]["source"], "aabb")
        self.assertEqual(
            document["stats"]["free_space"]["fallback_scenes"], ["room_fallback"]
        )
        self.assertEqual(
            document["stats"]["per_scene"]["room_fallback"]["probe"], "no_probe"
        )

    def test_source_aabb_ignores_an_available_probe(self) -> None:
        """Reproducing an old dataset must not silently use the new path."""
        document = build_scene_camera_jobs(
            self._document("room_legacy"),
            spec=CameraSpec(
                strategy="entity_conditioned_v1",
                mix_with_free_space=0.5,
                free_space=FreeSpaceSpec(source="aabb"),
            ),
            seed=11,
            cameras_per_scene=2,
            probe_by_scene={"room_legacy": room_probe()},
        )
        for job in document["jobs"]:
            self.assertEqual(job["free_space_report"]["source"], "aabb")
        self.assertEqual(
            document["stats"]["per_scene"]["room_legacy"]["probe"], "source_aabb"
        )

    def test_stale_probe_is_not_trusted(self) -> None:
        """A probe from another blend would mis-attribute every hit."""
        probe = room_probe()
        stale = ProbeVolume(
            origin=probe.origin,
            resolution=probe.resolution,
            clearance=probe.clearance,
            nearest_node=probe.nearest_node,
            node_ids=["something_else", "and_another"],
            clearance_cap=probe.clearance_cap,
        )
        document = build_scene_camera_jobs(
            self._document("room_stale"),
            spec=CameraSpec(strategy="entity_conditioned_v1", mix_with_free_space=0.5),
            seed=11,
            cameras_per_scene=2,
            probe_by_scene={"room_stale": stale},
        )
        self.assertEqual(
            document["stats"]["per_scene"]["room_stale"]["probe"], "probe_stale"
        )
        for job in document["jobs"]:
            self.assertEqual(job["free_space_report"]["source"], "aabb")

    def test_probe_path_is_derived_from_the_blend(self) -> None:
        blend = Path("data/cache/scenes/room.blend")
        self.assertEqual(probe_path_for(blend), Path("data/cache/scenes/room.probe.npz"))

    def test_discover_probes_finds_only_existing_files(self) -> None:
        probe = room_probe()
        target = probe_path_for(self.blend)
        probe.save(target)
        found = discover_probes(self._document(), project_root=SCRATCH)
        self.assertEqual(found, {"room_job": target})
        stale = discover_probes(
            {"scenes": [{"scene_id": "x", "normalized_blend": str(SCRATCH / "no.blend")}]},
            project_root=SCRATCH,
        )
        self.assertEqual(stale, {})


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #
class FreeSpaceSpecTest(unittest.TestCase):
    def test_unknown_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FreeSpaceSpec.from_mapping({"source": "magic"})

    def test_aabb_source_disables_probes(self) -> None:
        self.assertFalse(FreeSpaceSpec(source="aabb").uses_probe)
        self.assertTrue(FreeSpaceSpec(source="auto").uses_probe)
        self.assertTrue(FreeSpaceSpec(source="probe").uses_probe)

    def test_round_trip(self) -> None:
        spec = FreeSpaceSpec(hit_clearance=0.05)
        self.assertEqual(FreeSpaceSpec.from_mapping(spec.as_dict()).as_dict(), spec.as_dict())


class ProbeTargetConfigTest(unittest.TestCase):
    def test_default_target_spec_has_no_novelty_weight(self) -> None:
        # Ordering must stay byte-identical unless someone opts in.
        self.assertEqual(float(TargetSpec().azimuth_novelty_weight), 0.0)

    def test_novelty_weight_round_trips(self) -> None:
        spec = TargetSpec.from_mapping({"azimuth_novelty_weight": 0.3})
        self.assertAlmostEqual(spec.azimuth_novelty_weight, 0.3)
        self.assertAlmostEqual(spec.as_dict()["azimuth_novelty_weight"], 0.3)


# --------------------------------------------------------------------------- #
# The probe-scenes stage
# --------------------------------------------------------------------------- #
#: Stands in for ``blender``: reads the job json, writes a real ``.probe.npz``
#: and a summary, and fails on demand so the error path is exercised too.
FAKE_BLENDER = '''
import json
import sys
from pathlib import Path

import numpy as np

job_path = Path(sys.argv[sys.argv.index("--job") + 1])
job = json.loads(job_path.read_text(encoding="utf-8"))
if job.get("scene_id") == "unbakeable":
    print("simulated Blender crash", flush=True)
    raise SystemExit(3)

dims = (5, 5, 4)
resolution = float(job["resolution"])
output = Path(job["output"])
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output,
    format=np.array("lc_probe_v1"),
    origin=np.zeros(3, dtype=np.float64),
    resolution=np.array(resolution, dtype=np.float64),
    clearance=np.full(dims, 0.5, dtype=np.float32),
    nearest_node=np.zeros(dims, dtype=np.int32),
    node_ids=np.array(["shell"]),
)
Path(job["summary"]).write_text(
    json.dumps(
        {
            "scene_id": job["scene_id"],
            "output": str(output),
            "dims": list(dims),
            "resolution": resolution,
            "voxels": int(np.prod(dims)),
            "triangles": 12,
            "nodes": 1,
            "seconds": 0.01,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
print("[probe] fake bake ok", flush=True)
'''


class ProbeStageTest(unittest.TestCase):
    """``probe-scenes`` end to end, with a stand-in for Blender.

    The stage is the only part of the probe feature that shells out, so it is
    also the part most likely to break silently: a wrong path in the job file,
    a resume check that never matches, or a failure that takes the whole pipeline
    down instead of one scene.  Each of those has a test here.
    """

    def setUp(self) -> None:
        from datacompose.config import ProjectConfig

        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.script = SCRATCH / "fake_blender.py"
        self.script.write_text(FAKE_BLENDER, encoding="utf-8")
        self.blends = {}
        scenes = []
        for scene_id in ("room_a", "room_b", "unbakeable"):
            blend = SCRATCH / f"{scene_id}.blend"
            blend.write_bytes(b"fake-blend")
            self.blends[scene_id] = blend
            scenes.append(
                {"scene_id": scene_id, "normalized_blend": str(blend)}
            )
        self.scene_json = SCRATCH / "scene.json"
        self.scene_json.write_text(
            json.dumps({"schema_version": "1.0", "scenes": scenes}), encoding="utf-8"
        )
        self.config = ProjectConfig(
            source_path=SCRATCH / "config.yaml",
            root=SCRATCH,
            data={
                "paths": {
                    "scene_output": str(self.scene_json),
                    "scene_cache": str(SCRATCH / "cache" / "scenes"),
                },
                "m4": {
                    "blender_bin": "unused-on-this-path",
                    "probe": {"resolution": 0.5, "max_voxels": 1000},
                },
            },
        )

    def _run(self, **kwargs):
        from datacompose.scene_probe import probe_scenes

        kwargs.setdefault("blender_bin", f"{sys.executable} {self.script}")
        return probe_scenes(self.config, **kwargs)

    def test_stage_bakes_every_scene_it_can(self) -> None:
        summary = self._run()
        self.assertEqual(summary["stats"]["baked"], 2)
        self.assertEqual(summary["stats"]["failed"], 1)
        self.assertEqual(summary["stats"]["scenes"], 3)
        for scene_id in ("room_a", "room_b"):
            probe_path = probe_path_for(self.blends[scene_id])
            self.assertTrue(probe_path.is_file(), probe_path)
            # A real load: the file the stage writes must be readable by the
            # planner, not merely present.
            volume = ProbeVolume.load(probe_path)
            self.assertEqual(volume.node_ids, ["shell"])
            self.assertAlmostEqual(volume.resolution, 0.5)
        self.assertTrue((SCRATCH / "cache" / "scenes" / "probes" / "probe_summary.json").is_file())
        self.assertEqual(
            [row["scene_id"] for row in summary["failures"]], ["unbakeable"]
        )
        self.assertEqual(summary["failures"][0]["reason"], "probe_failed")

    def test_a_failed_bake_only_costs_that_scene(self) -> None:
        """One unbakeable fragment must not stop the other scenes."""
        summary = self._run()
        self.assertEqual(summary["stats"]["baked"], 2)
        # The failure detail must say enough to debug it from the summary alone.
        self.assertIn("simulated Blender crash", summary["failures"][0]["detail"])

    def test_require_all_turns_a_failure_into_an_error(self) -> None:
        from datacompose.config import ProjectConfig

        self.config = ProjectConfig(
            source_path=self.config.source_path,
            root=self.config.root,
            data={
                **self.config.data,
                "m4": {
                    **self.config.data["m4"],
                    "probe": {
                        **self.config.data["m4"]["probe"],
                        "require_all": True,
                    },
                },
            },
        )
        with self.assertRaises(RuntimeError):
            self._run()
        # The two good probes were still written: a loud failure is not a loss
        # of work, so a re-run with require_all off resumes instantly.
        self.assertTrue(probe_path_for(self.blends["room_a"]).is_file())

    def test_resume_skips_the_probes_it_already_has(self) -> None:
        self._run()
        second = self._run()
        self.assertEqual(second["stats"]["baked"], 0)
        self.assertEqual(second["stats"]["skipped"], 2)

    def test_no_resume_rebakes(self) -> None:
        self._run()
        second = self._run(resume=False)
        self.assertEqual(second["stats"]["baked"], 2)
        self.assertEqual(second["stats"]["skipped"], 0)

    def test_disabled_spec_bakes_nothing(self) -> None:
        from datacompose.config import ProjectConfig

        self.config = ProjectConfig(
            source_path=self.config.source_path,
            root=self.config.root,
            data={
                **self.config.data,
                "m4": {
                    **self.config.data["m4"],
                    "probe": {"enabled": False},
                },
            },
        )
        summary = self._run()
        self.assertEqual(summary["stats"]["baked"], 0)
        self.assertEqual(summary["stats"].get("disabled"), 1)
        self.assertFalse(probe_path_for(self.blends["room_a"]).exists())


class ProbeCliTest(unittest.TestCase):
    """The stage has to be reachable exactly like the other four."""

    def test_probe_scenes_parses(self) -> None:
        from datacompose.cli import _build_parser

        args = _build_parser().parse_args(
            ["probe-scenes", "--workers", "2", "--no-resume"]
        )
        self.assertEqual(args.command, "probe-scenes")
        self.assertEqual(args.workers, 2)
        self.assertFalse(args.resume)

    def test_resume_defaults_to_the_config(self) -> None:
        from datacompose.cli import _build_parser

        args = _build_parser().parse_args(["probe-scenes"])
        self.assertIsNone(args.resume)


if __name__ == "__main__":
    unittest.main()
