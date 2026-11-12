"""Static contract test: the planner's job must contain what the renderer reads.

``blender/render_scene.py`` only runs inside Blender, so it cannot be exercised
here.  Instead this test parses both sides:

* which literal keys the renderer subscribes from ``job`` and ``runtime``;
* which keys ``scene_cameras.build_camera_job`` produces, and which keys
  ``render_scenes.render_scenes`` puts into the worker snapshot.

A field renamed on one side only fails here instead of after a Blender run.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose.scene_cameras import CameraSpec, build_camera_job  # noqa: E402
from datacompose.tests.test_scene_cameras import room_scene  # noqa: E402


RENDERER = PACKAGE_DIR / "blender" / "render_scene.py"
SCHEDULER = PACKAGE_DIR / "render_scenes.py"


def _literal_subscript_paths(source: str, *, base_names: set[str]) -> set[tuple[str, ...]]:
    """Dotted literal subscript paths such as ``('camera', 'forward')``.

    Only chains rooted at one of ``base_names`` and made purely of string
    literals are reported; computed keys are skipped on purpose.
    """
    tree = ast.parse(source)
    paths: set[tuple[str, ...]] = set()

    def walk(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                # Outermost subscript is visited first, so prepend this key.
                walk(node.value, (key.value,) + prefix)
            # A computed key breaks the literal chain; stop here.
            return
        if isinstance(node, ast.Name):
            if node.id in base_names and prefix:
                paths.add(prefix)
            return
        if prefix:
            # The chain was interrupted (call/attribute), so it is not a plain
            # path rooted directly at the base name: drop it.
            return
        for child in ast.iter_child_nodes(node):
            walk(child, ())

    walk(tree, ())
    return paths


def _job_paths() -> set[tuple[str, ...]]:
    source = RENDERER.read_text(encoding="utf-8")
    # `job` and the aliases the renderer binds to its sub-blocks.
    return _literal_subscript_paths(source, base_names={"job"})


def _runtime_paths() -> set[tuple[str, ...]]:
    source = RENDERER.read_text(encoding="utf-8")
    return _literal_subscript_paths(source, base_names={"runtime"})


def _key_exists(payload: dict, path: tuple[str, ...]) -> bool:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _snapshot_keys() -> set[str]:
    """Keys of the ``worker_runtime`` dict literal in the scheduler."""
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "worker_runtime"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                return {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
    raise AssertionError("worker_runtime dict literal not found in render_scenes.py")


class RenderContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = room_scene()
        self.spec = CameraSpec()
        from datacompose.scene_cameras import sample_camera_poses

        result = sample_camera_poses(self.scene, self.spec, seed=3, count=1)
        self.assertGreaterEqual(len(result["accepted"]), 1, "test scene produced no camera")
        self.job = build_camera_job(
            self.scene,
            result["accepted"][0],
            spec=self.spec,
            seed=3,
            index=0,
            base_scene_blend=self.scene["normalized_blend"],
            base_scene_digest=self.scene["source_digest"],
            lighting_profile={"name": "scene_native_v1"},
            fixture_entity_ids=["room_a:entity:table"],
            license_block={"decision": "unverified"},
        )

    def test_renderer_job_fields_are_produced_by_the_planner(self) -> None:
        missing = sorted(
            ".".join(path) for path in _job_paths() if not _key_exists(self.job, path)
        )
        self.assertEqual(missing, [], f"renderer reads fields the planner does not emit: {missing}")

    def test_renderer_runtime_fields_are_written_by_the_scheduler(self) -> None:
        written = _snapshot_keys()
        read = {
            path[0]
            for path in _runtime_paths()
            if path and path[0] not in {"get", "items"} and not path[0].isdigit()
        }
        # Keys the scheduler must provide; `camera_gate` is nested but read whole.
        required = read | {"project_root", "jobs_path", "output_root", "worker_index", "worker_count"}
        self.assertTrue(
            required.issubset(written),
            f"scheduler snapshot is missing {sorted(required - written)}",
        )

    def test_camera_gate_keys_match_the_renderer_defaults(self) -> None:
        """Every threshold the pose validator reads must have a scheduler default."""
        source = RENDERER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        validator = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validate_camera_pose"
        )
        # The validator reads thresholds via `config.get("<name>", default)`.
        read = {
            node.args[0].value
            for node in ast.walk(validator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        self.assertTrue(read, "no gate thresholds found in _validate_camera_pose")

        scheduler = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
        builder = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef) and node.name == "_camera_gate"
        )
        provided = {
            node.args[0].value
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        missing = sorted(read - provided)
        self.assertEqual(missing, [], f"camera gate defaults are missing: {missing}")


if __name__ == "__main__":
    unittest.main()
