"""Regression tests for the planner→renderer job contract.

Covers the bug that made an entire batch fail with
``base scene blend digest changed after camera planning``: the planner used to
record the *source FBX* digest where the renderer expects the *blend* digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import unittest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose.scene_cameras import (  # noqa: E402
    CameraSpec,
    build_scene_camera_jobs,
    build_camera_job,
    sample_camera_poses,
)
from datacompose.tests.test_scene_cameras import room_scene  # noqa: E402


SCRATCH = PACKAGE_DIR / ".test-tmp"


def scratch_dir(name: str) -> Path:
    """Package-local scratch dir: OS temp may be unwritable in sandboxes."""
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def tearDownModule() -> None:
    """Leave no scratch files behind after a run."""
    shutil.rmtree(SCRATCH, ignore_errors=True)


class BlendDigestTest(unittest.TestCase):
    def test_job_digest_is_the_blend_digest_not_the_source_digest(self) -> None:
        root = scratch_dir("blend_digest")
        blend = root / "bistro_interior.blend"
        blend.write_bytes(b"fake-blend-payload")
        scene = room_scene("bistro_interior")
        scene["normalized_blend"] = str(blend)
        scene["source_digest"] = "sha256:" + "f" * 64  # deliberately different (FBX)

        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(),
            seed=20260810,
            cameras_per_scene=2,
        )

        self.assertGreaterEqual(len(document["jobs"]), 1)
        expected = sha256_of(blend)
        for job in document["jobs"]:
            self.assertEqual(job["base_scene_digest"], expected)
            self.assertEqual(job["lineage"]["blend_digest"], expected)
            # Provenance of the *source* asset is still recorded separately.
            self.assertEqual(job["lineage"]["source_scene_digest"], scene["source_digest"])

    def test_relative_blend_is_resolved_against_project_root_not_cwd(self) -> None:
        # ``scene.json`` records ``normalized_blend`` relative to the project
        # root, so the planner must not resolve it against the process CWD.
        root = scratch_dir("project_root")
        (root / "scenes").mkdir(parents=True, exist_ok=True)
        blend = root / "scenes" / "rel.blend"
        blend.write_bytes(b"root-payload")
        scene = room_scene("relative_blend")
        scene["normalized_blend"] = "scenes/rel.blend"

        other = scratch_dir("other_cwd")
        previous = Path.cwd()
        os.chdir(other)
        try:
            document = build_scene_camera_jobs(
                {"schema_version": "1.0", "scenes": [scene]},
                spec=CameraSpec(),
                seed=20260810,
                cameras_per_scene=2,
                project_root=root,
            )
        finally:
            os.chdir(previous)

        self.assertGreaterEqual(len(document["jobs"]), 1)
        expected = sha256_of(blend)
        for job in document["jobs"]:
            self.assertEqual(job["base_scene_digest"], expected)
            self.assertEqual(job["lineage"]["blend_digest"], expected)

    def test_missing_blend_is_rejected_instead_of_producing_a_broken_job(self) -> None:
        scene = room_scene("missing")
        scene["normalized_blend"] = str(scratch_dir("blend_missing") / "nope.blend")

        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(),
            seed=1,
            cameras_per_scene=1,
            require_all_scenes=False,
        )
        self.assertEqual(document["jobs"], [])
        self.assertEqual(document["rejects"][0]["reason"], "missing_base_scene_blend")

    def test_empty_blend_path_is_rejected(self) -> None:
        scene = room_scene("empty_path")
        scene["normalized_blend"] = ""

        document = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]},
            spec=CameraSpec(),
            seed=1,
            cameras_per_scene=1,
            require_all_scenes=False,
        )
        self.assertEqual(document["jobs"], [])
        self.assertEqual(document["rejects"][0]["reason"], "missing_normalized_blend_path")

    def test_free_space_report_exposes_ray_count(self) -> None:
        scene = room_scene("rays")
        result = sample_camera_poses(scene, CameraSpec(), seed=2, count=1)
        job = build_camera_job(
            scene,
            result["accepted"][0],
            spec=CameraSpec(),
            seed=2,
            index=0,
            base_scene_blend="x.blend",
            base_scene_digest="sha256:" + "0" * 64,
            lighting_profile={"name": "scene_native_v1"},
            source_scene_digest="sha256:" + "1" * 64,
        )
        self.assertGreater(job["free_space_report"]["ray_count"], 0)
        self.assertNotIn("node_count", job["free_space_report"])

    def test_job_ids_are_stable_across_runs(self) -> None:
        scene = room_scene("stable")
        blend = scratch_dir("blend_stable") / "s.blend"
        blend.write_bytes(b"payload")
        scene["normalized_blend"] = str(blend)
        first = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]}, spec=CameraSpec(), seed=7, cameras_per_scene=2
        )
        second = build_scene_camera_jobs(
            {"schema_version": "1.0", "scenes": [scene]}, spec=CameraSpec(), seed=7, cameras_per_scene=2
        )
        self.assertEqual(
            [job["job_id"] for job in first["jobs"]], [job["job_id"] for job in second["jobs"]]
        )
        self.assertTrue(all(job["job_id"].startswith("scene_camera_") for job in first["jobs"]))
        # job ids must not depend on the machine path
        self.assertEqual(
            json.dumps([job["job_id"] for job in first["jobs"]]),
            json.dumps([job["job_id"] for job in second["jobs"]]),
        )


if __name__ == "__main__":
    unittest.main()
