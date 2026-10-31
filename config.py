from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io_utils import stable_digest


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    root: Path
    data: dict[str, Any]

    @property
    def digest(self) -> str:
        return stable_digest(self.data)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' must be a mapping")
        return value

    def path(self, name: str) -> Path:
        paths = self.section("paths")
        if name not in paths:
            raise KeyError(f"Missing config path: paths.{name}")
        value = Path(os.path.expandvars(str(paths[name])))
        return value if value.is_absolute() else (self.root / value).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    source_path = Path(path).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top-level config must be a mapping: {source_path}")
    data = _expand(copy.deepcopy(raw))
    project = data.setdefault("project", {})
    root_value = Path(str(project.get("root", ".")))
    root = root_value if root_value.is_absolute() else source_path.parent / root_value
    return ProjectConfig(source_path=source_path, root=root.resolve(), data=data)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value

