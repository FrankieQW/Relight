"""Tests for the render scheduler's partial-output handling.

The batch-level clear exists because one rejected/failed pose per camera used to
leave one partial directory behind, and retrying a 64-camera batch then meant 64
manual ``clear-render-partial --job-id`` calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose.tests.test_launcher import _install_yaml_stub  # noqa: E402

# `config.py` imports PyYAML at module scope, so the stub has to exist first.
_install_yaml_stub()

from datacompose.config import ProjectConfig  # noqa: E402
from datacompose.render_scenes import clear_failed_partials  # noqa: E402

SCRATCH = PACKAGE_DIR / ".test-tmp"


def tearDownModule() -> None:
    """Leave no scratch files behind after a run."""
    shutil.rmtree(SCRATCH, ignore_errors=True)


def build_case(name: str) -> tuple[ProjectConfig, Path]:
    _install_yaml_stub()
    root = SCRATCH / name
    shutil.rmtree(root, ignore_errors=True)
    output = root / "dataset"
    components = output / "components"
    components.mkdir(parents=True)
    jobs = [{"job_id": f"scene_camera_{index:03d}"} for index in range(5)]
    # 000: completed render, stale partial must survive
    (components / "scene_camera_000").mkdir()
    (components / "scene_camera_000" / "metadata.json").write_text("{}", encoding="utf-8")
    (components / "scene_camera_000.partial").mkdir()
    (components / "scene_camera_000.partial" / "failure.json").write_text("{}", encoding="utf-8")
    # 001, 002: failed with evidence -> cleared
    for job_id in ("scene_camera_001", "scene_camera_002"):
        (components / f"{job_id}.partial").mkdir()
        (components / f"{job_id}.partial" / "failure.json").write_text("{}", encoding="utf-8")
    # 003: partial without failure evidence (possibly still running) -> kept
    (components / "scene_camera_003.partial").mkdir()
    # 004: no partial at all
    jobs_path = root / "jobs.jsonl"
    jobs_path.write_text("\n".join(json.dumps(row) for row in jobs), encoding="utf-8")
    config = ProjectConfig(
        source_path=root / "config.yaml",
        root=root,
        data={
            "paths": {
                "tokenlight_output": str(output),
                "render_jobs_output": str(jobs_path),
            }
        },
    )
    return config, components


class ClearFailedPartialsTest(unittest.TestCase):
    def test_only_evidenced_partials_of_manifest_jobs_are_cleared(self) -> None:
        config, components = build_case("clear_partials")
        result = clear_failed_partials(config)
        self.assertEqual(result["stats"]["cleared"], 2)
        self.assertEqual(result["cleared_job_ids"], ["scene_camera_001", "scene_camera_002"])
        reasons = {row["job_id"]: row["reason"] for row in result["kept"]}
        self.assertEqual(reasons["scene_camera_000"], "completed_output_exists")
        self.assertEqual(reasons["scene_camera_003"], "no_failure_evidence")
        self.assertEqual(
            sorted(path.name for path in components.glob("*.partial")),
            ["scene_camera_000.partial", "scene_camera_003.partial"],
        )

    def test_empty_manifest_is_rejected(self) -> None:
        config, _ = build_case("clear_empty")
        config.path("render_jobs_output").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            clear_failed_partials(config)

    def test_render_refuses_before_launching_when_a_partial_survives(self) -> None:
        """Without --clear-failed the batch must stop before any Blender process."""
        from datacompose.render_scenes import render_scenes

        config, _ = build_case("render_refuses")
        with patch("datacompose.render_scenes.subprocess.Popen") as popen:
            with self.assertRaises(FileExistsError):
                render_scenes(config)
        popen.assert_not_called()

    def test_clear_failed_lets_the_batch_start(self) -> None:
        from datacompose.render_scenes import render_scenes

        config, _ = build_case("render_clears")
        config.data.setdefault("m4", {})
        config.data["m4"].update(
            {
                "blender_bin": "/bin/true",
                "render_gpu_ids": [0],
                "render_workers": 1,
                "render": {"resolution": 64},
            }
        )

        class _Done:
            returncode = 0

            def wait(self) -> None:
                return None

        with patch("datacompose.render_scenes.subprocess.Popen", return_value=_Done()) as popen:
            with patch("datacompose.render_scenes.assert_no_existing_render_partials") as guard:
                summary = render_scenes(config, clear_failed=True, allow_partial=True)
        popen.assert_called_once()
        # Partials were cleared, then the guard ran against a clean tree.
        guard.assert_called_once()
        self.assertEqual(summary["cleared_partials"]["stats"]["cleared"], 2)


if __name__ == "__main__":
    unittest.main()
