"""Bootstrap that lets every stage run as a plain script.

The pipeline is a package, so the historical invocation is::

    python -m datacompose.cli prepare-scenes --config sceneonly.yaml

which forces the caller to be in the *parent* of the package directory and to
remember the module path.  The thin scripts next to this file (``prepare_scenes.py``,
``render_dataset.py``, ...) call :func:`run` instead, so this works from any
directory and from a renamed project folder::

    python prepare_scenes.py --blender-bin /opt/blender-4.5/blender

Two problems are solved here:

* **Package aliasing.** ``datacompose`` is imported by absolute name throughout
  the code base.  If the checkout directory has another name (for example
  ``datacompose-sceneonly``, which is not a valid module name), a plain
  ``import datacompose`` would fail.  :func:`ensure_package` registers this
  directory in ``sys.modules`` under the canonical name, so submodules resolve
  exactly as they do with ``python -m``.
* **Config default.** ``--config`` may be omitted when ``DATACOMPOSE_CONFIG``
  points at a YAML file; otherwise the bundled ``configs/default.yaml`` is used.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PACKAGE_NAME = "datacompose"
PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_ENV = "DATACOMPOSE_CONFIG"


def ensure_package() -> None:
    """Make ``import datacompose`` resolve to this directory, whatever its name."""
    if PACKAGE_NAME in sys.modules:
        return
    init_path = PACKAGE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"not a datacompose package directory: {PACKAGE_DIR}")
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create the package spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    parent = str(PACKAGE_DIR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def default_config() -> str:
    """``--config`` value used when the caller does not pass one."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        # Keep the caller's spelling: `str(Path(...))` would rewrite separators
        # on Windows and surprise anyone scripting with POSIX paths.
        return os.path.expanduser(override)
    return str(PACKAGE_DIR / "configs" / "default.yaml")


def run(command: str, argv: list[str] | None = None) -> None:
    """Run one pipeline stage as if it had been given on the module CLI."""
    ensure_package()
    from datacompose.cli import main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--config" or item.startswith("--config=") for item in arguments):
        arguments = ["--config", default_config(), *arguments]
    main([command, *arguments])


__all__ = ["CONFIG_ENV", "PACKAGE_DIR", "PACKAGE_NAME", "default_config", "ensure_package", "run"]
