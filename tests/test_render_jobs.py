from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from datacompose.config import ProjectConfig
from datacompose.io_utils import sha256_file
from datacompose.render_jobs import build_render_jobs


class RenderJobsTest(unittest.TestCase):
    def test_one_deterministic_job_per_object_scene_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "outputs"
            data.mkdir()
            blend = data / "scene.blend"
            blend.write_bytes(b"blend-fixture")
            object_path = data / "object.json"
            scene_path = data / "scene.json"
            annotation_path = data / "annotation.json"
            geometry_path = data / "geometry.json"
            object_document = self._object_document()
            self._write(object_path, object_document)
            scene_document = self._scene_document(blend.relative_to(root).as_posix())
            self._write(scene_path, scene_document)
            annotation_document = self._annotation_document(
                sha256_file(object_path), sha256_file(scene_path)
            )
            self._write(annotation_path, annotation_document)
            geometry_document = self._geometry_document(sha256_file(object_path))
            self._write(geometry_path, geometry_document)
            config = ProjectConfig(
                source_path=root / "config.yaml",
                root=root,
                data={
                    "project": {"schema_version": "1.0", "seed": 7},
                    "paths": {
                        "annotation_output": str(annotation_path),
                        "object_output": str(object_path),
                        "scene_output": str(scene_path),
                        "prepared_geometry_output": str(geometry_path),
                        "render_jobs_output": str(output / "jobs.jsonl"),
                        "render_jobs_summary": str(output / "jobs.json"),
                        "render_job_rejects": str(output / "rejects.jsonl"),
                    },
                    "m4": {
                        "relation_priority": ["replace", "place_on"],
                        "replacement_fill_ratio": 0.9,
                        "yaw_degrees": [0, 90],
                        "fixture_categories": ["lamp"],
                        "base_scene_license": {"decision": "allowed"},
                        "license_policy_version": "test-v1",
                    },
                },
            )
            first = build_render_jobs(config)
            second = build_render_jobs(config)
            self.assertEqual(len(first["jobs"]), 1)
            self.assertEqual(first["jobs"], second["jobs"])
            self.assertEqual(first["jobs"][0]["target"]["relation"], "replace")
            self.assertIn(
                first["jobs"][0]["target"]["entity_id"],
                {"scene:entity:table", "scene:entity:table-secondary"},
            )
            self.assertEqual(first["jobs"][0]["license"]["decision"], "allowed")
            self.assertEqual(first["jobs"][0]["camera"]["strategy"], "generated_target_visible")
            self.assertNotIn("name", first["jobs"][0]["camera"])
            self.assertEqual(first["jobs"][0]["base_scene_digest"], sha256_file(blend))
            render_job_digest = first["jobs"][0]["lineage"]["render_job_digest"]
            self.assertEqual(
                first["jobs"][0]["job_id"],
                "composition_" + render_job_digest.removeprefix("sha256:")[:20],
            )

            config.data["m4"]["render"] = {"samples": 8}
            changed_contract = build_render_jobs(config)
            self.assertNotEqual(
                first["jobs"][0]["job_id"], changed_contract["jobs"][0]["job_id"]
            )

    @staticmethod
    def _object_document() -> dict:
        return {
            "schema_version": "1.0",
            "generated_at": "2026-08-10T00:00:00Z",
            "generator_version": "test",
            "config_digest": "sha256:" + "1" * 64,
            "inventory_mode": "test",
            "object_root_env": "OBJECT_ROOT",
            "source_digests": {},
            "stats": {},
            "objects": [
                {
                    "uid": "object-1",
                    "primary_category": "Mug",
                    "primary_category_normalized": "mug",
                    "categories": ["Mug"],
                    "categories_normalized": ["mug"],
                    "inventory_path": "objects/object.glb",
                    "canonical_path": "objects/object.glb",
                    "shard": "objects",
                    "license_raw": "cc0",
                    "license": "CC0",
                    "source_uri": "https://example.invalid/object",
                    "embed_url": None,
                    "name": "Mug",
                    "description": None,
                    "tags": [],
                    "author": {},
                    "is_age_restricted": False,
                    "is_downloadable": True,
                    "glb_stats": None,
                    "thumbnails": [],
                    "metadata_status": "available",
                }
            ],
        }

    @staticmethod
    def _scene_document(blend_path: str) -> dict:
        entities = [
            {
                "entity_id": "scene:entity:table",
                "node_ids": ["scene:node:table"],
                "raw_label": "table",
                "category": "table",
                "category_confidence": 1.0,
                "grouping_confidence": 1.0,
                "grouping_method": "test",
                "replaceable": True,
                "support_surface": True,
                "obb_world": {
                    "center": [0.0, 0.0, 0.5],
                    "dimensions": [1.0, 1.0, 1.0],
                },
                "override_applied": False,
            },
            {
                "entity_id": "scene:entity:lamp",
                "node_ids": ["scene:node:lamp"],
                "raw_label": "lamp",
                "category": "lamp",
                "category_confidence": 1.0,
                "grouping_confidence": 1.0,
                "grouping_method": "test",
                "replaceable": False,
                "support_surface": False,
                "obb_world": {"center": [0.0, 0.0, 2.0], "dimensions": [0.2, 0.2, 0.4]},
                "override_applied": False,
            },
            {
                "entity_id": "scene:entity:table-secondary",
                "node_ids": ["scene:node:table-secondary"],
                "raw_label": "table secondary",
                "category": "table",
                "category_confidence": 1.0,
                "grouping_confidence": 1.0,
                "grouping_method": "test",
                "replaceable": True,
                "support_surface": True,
                "obb_world": {
                    "center": [2.0, 0.0, 0.5],
                    "dimensions": [1.0, 1.0, 1.0],
                },
                "override_applied": False,
            },
        ]
        return {
            "schema_version": "1.0",
            "generated_at": "2026-08-10T00:00:00Z",
            "generator_version": "test",
            "config_digest": "sha256:" + "2" * 64,
            "source_digests": {},
            "stats": {},
            "scenes": [
                {
                    "scene_id": "scene",
                    "source_fbx": "scene.fbx",
                    "source_digest": "sha256:" + "3" * 64,
                    "normalized_blend": blend_path,
                    "units": "meter",
                    "up_axis": "+Z",
                    "config_digest": "sha256:" + "2" * 64,
                    "nodes": [],
                    "entities": entities,
                    "stats": {},
                }
            ],
        }

    @staticmethod
    def _annotation_document(object_digest: str, scene_digest: str) -> dict:
        return {
            "schema_version": "1.0",
            "generated_at": "2026-08-10T00:00:00Z",
            "generator_version": "test",
            "config_digest": "sha256:" + "4" * 64,
            "object_digest": object_digest,
            "scene_digest": scene_digest,
            "model": {
                "name": "test",
                "base_url": "http://localhost",
                "prompt_version": "v1",
                "temperature": 0.0,
                "thinking": False,
            },
            "class_rules": [
                {
                    "object_category": "mug",
                    "scene_category": "table",
                    "can_place_on": True,
                    "can_replace": True,
                    "confidence": 0.95,
                    "reason": "test",
                    "cache_key": "test-rule",
                }
            ],
            "targets_by_object_category": {
                "mug": {
                    "place_on_entity_ids": [
                        "scene:entity:table-secondary",
                        "scene:entity:table",
                    ],
                    "replace_entity_ids": [
                        "scene:entity:table-secondary",
                        "scene:entity:table",
                    ],
                }
            },
            "object_index": {"object-1": "mug"},
            "unresolved_pairs": [],
            "stats": {},
        }

    @staticmethod
    def _geometry_document(object_digest: str) -> dict:
        geometry = {
            "object_uid": "object-1",
            "object_category": "mug",
            "asset_path": "objects/object.glb",
            "asset_digest": "sha256:" + "5" * 64,
            "target_dimensions": [0.1, 0.1, 0.12],
            "up_axis": "+Z",
            "front_axis": "-Y",
            "contact_axis": "-Z",
            "fit_mode": "uniform_fit",
            "config_source": "category_default",
            "license": {"name": "CC0", "decision": "allowed"},
        }
        return {
            "schema_version": "1.0",
            "generated_at": "2026-08-10T00:00:00Z",
            "generator_version": "test",
            "config_digest": "sha256:" + "6" * 64,
            "object_digest": object_digest,
            "geometry_config_digests": {},
            "object_root_env": "OBJECT_ROOT",
            "geometries": [geometry],
            "quarantine": [],
            "stats": {},
        }

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
