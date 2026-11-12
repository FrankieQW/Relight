from __future__ import annotations

import contextlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import types
import unittest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_yaml_stub() -> None:
    """Allow config parsing when PyYAML is absent (JSON is valid YAML)."""
    try:
        import yaml  # noqa: F401

        return
    except ImportError:
        pass
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda text: json.loads(text)  # type: ignore[attr-defined]
    sys.modules["yaml"] = stub


_SCRATCH_COUNTER = itertools.count()


@contextlib.contextmanager
def scratch():
    """Package-local scratch directory with best-effort cleanup.

    Plain ``mkdir`` is used instead of ``tempfile.mkdtemp`` on purpose: some
    sandboxes let a process create a private temp directory but then deny every
    write inside it, which would fail the test for reasons unrelated to the code.
    """
    root = PACKAGE_DIR / ".test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run-{os.getpid()}-{next(_SCRATCH_COUNTER)}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_launcher(path: Path, name: str = "_launcher_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTest(unittest.TestCase):
    def test_renamed_directory_is_registered_as_datacompose(self) -> None:
        """The checkout folder may be called anything; the package name is fixed."""
        with scratch() as temporary:
            renamed = Path(temporary) / "datacompose-sceneonly"
            renamed.mkdir()
            shutil.copyfile(PACKAGE_DIR / "__init__.py", renamed / "__init__.py")
            shutil.copyfile(PACKAGE_DIR / "_launcher.py", renamed / "_launcher.py")
            (renamed / "probe.py").write_text("VALUE = 7\n", encoding="utf-8")

            launcher = _load_launcher(renamed / "_launcher.py", "_launcher_renamed")
            saved = {
                name: sys.modules.pop(name)
                for name in list(sys.modules)
                if name == "datacompose" or name.startswith("datacompose.")
            }
            try:
                launcher.ensure_package()
                self.assertIn("datacompose", sys.modules)
                self.assertEqual(
                    Path(sys.modules["datacompose"].__path__[0]).resolve(), renamed.resolve()
                )
                import datacompose.probe  # noqa: F401,PLC0415 - import after aliasing

                self.assertEqual(datacompose.probe.VALUE, 7)
            finally:
                for name in [n for n in sys.modules if n == "datacompose" or n.startswith("datacompose.")]:
                    if name not in saved:
                        sys.modules.pop(name, None)
                sys.modules.update(saved)

    def test_default_config_prefers_environment_variable(self) -> None:
        launcher = _load_launcher(PACKAGE_DIR / "_launcher.py", "_launcher_config")
        previous = os.environ.get(launcher.CONFIG_ENV)
        try:
            os.environ.pop(launcher.CONFIG_ENV, None)
            self.assertTrue(launcher.default_config().endswith(os.path.join("configs", "default.yaml")))
            os.environ[launcher.CONFIG_ENV] = "/data/frankie/sceneonly.yaml"
            self.assertEqual(launcher.default_config(), "/data/frankie/sceneonly.yaml")
        finally:
            if previous is None:
                os.environ.pop(launcher.CONFIG_ENV, None)
            else:
                os.environ[launcher.CONFIG_ENV] = previous

    def test_run_injects_config_only_when_missing(self) -> None:
        launcher = _load_launcher(PACKAGE_DIR / "_launcher.py", "_launcher_run")
        _install_yaml_stub()
        from datacompose import cli  # noqa: PLC0415 - needs the stub in place

        calls: list[list[str]] = []
        original = cli.main
        cli.main = lambda argv=None: calls.append(list(argv or []))  # type: ignore[assignment]
        previous = os.environ.get(launcher.CONFIG_ENV)
        try:
            os.environ[launcher.CONFIG_ENV] = "/tmp/x.yaml"
            launcher.run("prepare-scenes", ["--blender-bin", "/opt/blender"])
            launcher.run("prepare-scenes", ["--config", "custom.yaml"])
            launcher.run("prepare-scenes", ["--config=custom.yaml"])
        finally:
            cli.main = original  # type: ignore[assignment]
            if previous is None:
                os.environ.pop(launcher.CONFIG_ENV, None)
            else:
                os.environ[launcher.CONFIG_ENV] = previous
        self.assertEqual(
            calls[0], ["prepare-scenes", "--config", "/tmp/x.yaml", "--blender-bin", "/opt/blender"]
        )
        self.assertEqual(calls[1], ["prepare-scenes", "--config", "custom.yaml"])
        self.assertEqual(calls[2], ["prepare-scenes", "--config=custom.yaml"])

    def test_launcher_scripts_exist_for_every_stage(self) -> None:
        expected = {
            "prepare_scenes.py": "prepare-scenes",
            "plan_scene_cameras.py": "plan-scene-cameras",
            "render_dataset.py": "render",
            "clear_render_partial.py": "clear-render-partial",
        }
        for name, command in expected.items():
            path = PACKAGE_DIR / name
            self.assertTrue(path.is_file(), name)
            text = path.read_text(encoding="utf-8")
            self.assertIn(f'run("{command}")', text, name)


class PlanSceneCamerasCommandTest(unittest.TestCase):
    """End-to-end: config -> scene.json -> scene_camera_jobs.jsonl, no Blender."""

    def test_cli_plans_cameras_for_a_scene(self) -> None:
        _install_yaml_stub()
        from datacompose import cli  # noqa: PLC0415
        from datacompose.tests.test_scene_cameras import room_scene  # noqa: PLC0415

        with scratch() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            scene_path.write_text(
                json.dumps({"schema_version": "1.0", "scenes": [room_scene("room_a")]}),
                encoding="utf-8",
            )
            config_path = root / "sceneonly.json"  # JSON is valid YAML
            config_path.write_text(
                json.dumps(
                    {
                        "project": {"root": str(root), "seed": 20260810},
                        "paths": {"scene_output": str(scene_path)},
                    }
                ),
                encoding="utf-8",
            )
            jobs_path = root / "scene_camera_jobs.jsonl"
            cli.main(
                [
                    "plan-scene-cameras",
                    "--config",
                    str(config_path),
                    "--cameras-per-scene",
                    "3",
                    "--output",
                    str(jobs_path),
                ]
            )
            rows = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), 3)
            summary = json.loads(jobs_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stats"]["jobs"], 3)
            self.assertEqual(summary["stats"]["scenes"], 1)
            for row in rows:
                self.assertEqual(row["scene_id"], "room_a")
                self.assertEqual(row["camera"]["strategy"], "scene_free_space_v1")
                self.assertGreaterEqual(row["camera"]["minimum_hit_depth"], 0.30 - 1e-9)


if __name__ == "__main__":
    unittest.main()
