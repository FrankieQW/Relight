from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from datacompose.render_recovery import (
    assert_no_existing_render_partials,
    clear_failed_render_partial,
    prepare_render_partial,
)


class RenderRecoveryTest(unittest.TestCase):
    def test_existing_failure_is_preserved_and_locatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            partial = output_root / "components" / "job-a.partial"
            partial.mkdir(parents=True)
            failure = partial / "failure.json"
            evidence = {"job_id": "job-a", "traceback": "original traceback"}
            failure.write_text(json.dumps(evidence), encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, re.escape(str(failure))):
                assert_no_existing_render_partials(output_root, [{"job_id": "job-a"}])

            with self.assertRaisesRegex(FileExistsError, re.escape(str(failure))):
                prepare_render_partial(output_root, "job-a")

            self.assertTrue(partial.is_dir())
            self.assertEqual(json.loads(failure.read_text(encoding="utf-8")), evidence)

    def test_explicit_cleanup_is_limited_to_one_manifest_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            jobs = [{"job_id": "job-a"}, {"job_id": "job-b"}]
            for job_id in ("job-a", "job-b"):
                partial = output_root / "components" / f"{job_id}.partial"
                partial.mkdir(parents=True)
                (partial / "failure.json").write_text("{}", encoding="utf-8")

            result = clear_failed_render_partial(output_root, jobs, "job-a")

            self.assertEqual(result["job_id"], "job-a")
            self.assertFalse((output_root / "components" / "job-a.partial").exists())
            other_failure = output_root / "components" / "job-b.partial" / "failure.json"
            self.assertTrue(other_failure.is_file())
            recreated = prepare_render_partial(output_root, "job-a")
            self.assertTrue(recreated.is_dir())

    def test_cleanup_refuses_partial_without_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            partial = output_root / "components" / "job-a.partial"
            partial.mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "may still be active"):
                clear_failed_render_partial(output_root, [{"job_id": "job-a"}], "job-a")

            self.assertTrue(partial.is_dir())


if __name__ == "__main__":
    unittest.main()
