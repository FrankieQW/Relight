from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datacompose.config import ProjectConfig
from datacompose.render_compositions import render_compositions


ROOT = Path(__file__).resolve().parents[1]


class _CompletedProcess:
    returncode = 0

    def wait(self) -> None:
        return None


class RenderCompositionsTest(unittest.TestCase):
    def test_summary_uses_manifest_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_path = root / "jobs.jsonl"
            output_root = root / "dataset"
            object_root = root / "objects"
            object_root.mkdir()
            job_id = "composition_test"
            jobs_path.write_text(json.dumps({"job_id": job_id}) + "\n", encoding="utf-8")
            metadata = output_root / "components" / job_id / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("{}\n", encoding="utf-8")
            config = ProjectConfig(
                source_path=root / "config.yaml",
                root=ROOT,
                data={
                    "paths": {
                        "render_jobs_output": str(jobs_path),
                        "tokenlight_output": str(output_root),
                    },
                    "scene": {"blender_bin": "blender"},
                    "m4": {
                        "object_root_env": "TEST_OBJECT_ROOT",
                        "render_gpu_ids": [0],
                        "render_workers": 1,
                    },
                },
            )
            with (
                patch.dict(os.environ, {"TEST_OBJECT_ROOT": str(object_root)}),
                patch("datacompose.render_compositions.subprocess.Popen", return_value=_CompletedProcess()),
            ):
                summary = render_compositions(config)
            self.assertEqual(summary["jobs"], 1)
            self.assertEqual(summary["rendered_or_reused"], 1)
            self.assertEqual(summary["missing_job_outputs"], [])

    def test_hdri_root_is_resolved_and_written_to_worker_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_path = root / "jobs.jsonl"
            output_root = root / "dataset"
            object_root = root / "objects"
            hdri_root = root / "hdris"
            object_root.mkdir()
            hdri_root.mkdir()
            (hdri_root / "studio.HDR").write_bytes(b"test")
            job_id = "composition_test"
            jobs_path.write_text(json.dumps({"job_id": job_id}) + "\n", encoding="utf-8")
            metadata = output_root / "components" / job_id / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("{}\n", encoding="utf-8")
            config = ProjectConfig(
                source_path=root / "config.yaml",
                root=root,
                data={
                    "paths": {
                        "render_jobs_output": str(jobs_path),
                        "tokenlight_output": str(output_root),
                    },
                    "scene": {"blender_bin": "blender"},
                    "m4": {
                        "object_root_env": "TEST_OBJECT_ROOT",
                        "render_gpu_ids": [0],
                        "render_workers": 1,
                        "render": {"hdri_root": "hdris"},
                    },
                },
            )
            with (
                patch.dict(os.environ, {"TEST_OBJECT_ROOT": str(object_root)}),
                patch("datacompose.render_compositions.subprocess.Popen", return_value=_CompletedProcess()),
            ):
                render_compositions(config)
            runtime = json.loads(
                (output_root / "runtime" / "composition_worker_000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(runtime["render"]["hdri_root"], str(hdri_root.resolve()))

    def test_empty_hdri_root_fails_before_blender_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_path = root / "jobs.jsonl"
            output_root = root / "dataset"
            object_root = root / "objects"
            hdri_root = root / "hdris"
            object_root.mkdir()
            hdri_root.mkdir()
            jobs_path.write_text(json.dumps({"job_id": "composition_test"}) + "\n", encoding="utf-8")
            config = ProjectConfig(
                source_path=root / "config.yaml",
                root=root,
                data={
                    "paths": {
                        "render_jobs_output": str(jobs_path),
                        "tokenlight_output": str(output_root),
                    },
                    "scene": {"blender_bin": "blender"},
                    "m4": {
                        "object_root_env": "TEST_OBJECT_ROOT",
                        "render_gpu_ids": [0],
                        "render_workers": 1,
                        "render": {"hdri_root": str(hdri_root)},
                    },
                },
            )
            with patch.dict(os.environ, {"TEST_OBJECT_ROOT": str(object_root)}):
                with self.assertRaisesRegex(ValueError, "contains no .hdr or .exr"):
                    render_compositions(config)


if __name__ == "__main__":
    unittest.main()
