from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .lighting import LightingSchema


SPLIT_SEED_OFFSETS = {"train": 0, "validation": 10_000_000, "test": 20_000_000}


def _range(config: dict[str, Any], name: str) -> tuple[float, float]:
    values = config["data"]["ranges"][name]
    return float(values[0]), float(values[1])


class TokenLightKontextDataset(Dataset):
    """Build deterministic source/target/lighting triples from TokenLight renders."""

    def __init__(self, config: dict[str, Any], split: str):
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError(f"unsupported split: {split}")
        self.config = config
        self.split = split
        self.root = Path(config["paths"]["dataset_root"]).expanduser()
        manifest_value = config["paths"].get(f"{split}_manifest")
        if not manifest_value:
            raise ValueError(f"paths.{split}_manifest is required")
        self.manifest_path = Path(manifest_value).expanduser()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.scenes = [json.loads(line) for line in handle if line.strip()]
        if not self.scenes:
            raise ValueError(f"manifest contains no scenes: {self.manifest_path}")

        data = config["data"]
        self.resolution = int(data["resolution"])
        self.exposure = float(data["exposure"])
        self.samples_per_scene = int(data["samples_per_scene"][split])
        self.seed = int(data["seed"]) + SPLIT_SEED_OFFSETS[split]
        self.tasks = tuple(data["tasks"])
        self.task_probabilities = dict(data["task_probabilities"])
        self.max_lights = int(config["model"]["max_lights"])
        self.fixture_mask_enabled = bool(config["model"]["fixture_mask_enabled"])
        self.schema = LightingSchema(self.max_lights)
        self.ambient_range = _range(config, "ambient_scale")
        self.color_range = _range(config, "light_color")
        self.intensity_range = _range(config, "light_intensity")
        self.fixture_intensity_range = _range(config, "fixture_intensity")
        self.fixture_transition_range = _range(config, "fixture_transition")

    def __len__(self) -> int:
        return len(self.scenes) * self.samples_per_scene

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        sample_seed = self.seed + index
        rng = np.random.default_rng(sample_seed)
        scene = self.scenes[index % len(self.scenes)]
        available = [task for task in self.tasks if self._supports(scene, task)]
        if not available:
            raise ValueError(f"scene {scene.get('id')} supports none of the enabled tasks")
        weights = np.asarray(
            [float(self.task_probabilities.get(task, 0.0)) for task in available], dtype=np.float64
        )
        if weights.sum() <= 0:
            raise ValueError(f"scene {scene.get('id')} has zero probability across available tasks")
        task = str(rng.choice(available, p=weights / weights.sum()))

        ambient = self._read_linear(scene["ambient"])
        dark = self._read_linear(scene["dark"]) if scene.get("dark") else np.zeros_like(ambient)
        source, target, control = self._compose(scene, task, ambient, dark, rng)
        lighting = self.schema.pack(task, control)
        fixture_present = task == "in_scene_light" and self.fixture_mask_enabled
        if fixture_present:
            mask_value = control.get("fixture_mask")
            if not mask_value:
                raise ValueError(f"in_scene_light sample has no fixture mask: {scene.get('id')}")
            fixture_mask = self._read_mask(mask_value)
            if fixture_mask.shape != ambient.shape[:2]:
                raise ValueError(
                    "fixture mask shape differs from ambient image: "
                    f"{fixture_mask.shape} != {ambient.shape[:2]}"
                )
        else:
            fixture_mask = np.zeros(ambient.shape[:2], dtype=np.float32)
        prepared_mask = self._prepare_mask(fixture_mask)
        if fixture_present and float(prepared_mask.max()) <= 0.0:
            raise ValueError(f"in_scene_light fixture mask is empty: {scene.get('id')}")
        return {
            "condition_pixel_values": torch.from_numpy(self._prepare_image(source)).permute(2, 0, 1),
            "target_pixel_values": torch.from_numpy(self._prepare_image(target)).permute(2, 0, 1),
            "lighting_values": torch.from_numpy(lighting.values.copy()),
            "lighting_known": torch.from_numpy(lighting.known.copy()),
            "lighting_valid": torch.from_numpy(lighting.valid.copy()),
            "lighting_task_id": torch.tensor(lighting.task_id, dtype=torch.long),
            "fixture_mask": torch.from_numpy(prepared_mask)[None],
            "fixture_present": torch.tensor(fixture_present, dtype=torch.bool),
            "control": control,
            "task_name": task,
            "scene_id": str(scene["id"]),
            "sample_index": index,
            "sample_seed": sample_seed,
        }

    def _compose(
        self,
        scene: dict[str, Any],
        task: str,
        ambient: np.ndarray,
        dark: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if task == "ambient_scale":
            scale = float(rng.uniform(*self.ambient_range))
            return ambient, dark + (ambient - dark) * scale, {"scale": scale}

        if task == "global_diffuse":
            source_index, target_index = rng.choice(len(scene["diffuse"]), size=2, replace=False)
            source_component = scene["diffuse"][int(source_index)]
            target_component = scene["diffuse"][int(target_index)]
            source = ambient + self._read_linear(source_component["path"]) - dark
            target = ambient + self._read_linear(target_component["path"]) - dark
            source_level = float(source_component["level"])
            target_level = float(target_component["level"])
            return source, target, {
                "source_level": source_level,
                "target_level": target_level,
                "delta": target_level - source_level,
            }

        if task == "add_light":
            component_count = min(len(scene["point_lights"]), self.max_lights)
            light_count = int(rng.integers(1, component_count + 1))
            selected = rng.choice(len(scene["point_lights"]), size=light_count, replace=False)
            target = ambient.copy()
            lights: list[dict[str, float]] = []
            for component_index in selected:
                component = scene["point_lights"][int(component_index)]
                color = rng.uniform(*self.color_range, size=3).astype(np.float32)
                intensity = float(rng.uniform(*self.intensity_range))
                contribution = np.maximum(self._read_linear(component["path"]) - dark, 0.0)
                target += contribution * color[None, None, :] * intensity
                position = component["position"]
                lights.append(
                    {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "r": float(color[0]),
                        "g": float(color[1]),
                        "b": float(color[2]),
                        "intensity": intensity,
                        "softness": float(component.get("diffuse", 0.0)),
                    }
                )
            return ambient, target, {"lights": lights}

        fixtures = self._usable_fixtures(scene)
        fixture = fixtures[int(rng.integers(len(fixtures)))]
        color = rng.uniform(*self.color_range, size=3).astype(np.float32)
        intensity = float(rng.uniform(*self.fixture_intensity_range))
        transition = float(rng.uniform(*self.fixture_transition_range))
        contribution = np.maximum(self._read_linear(fixture.get("path") or fixture.get("on")) - dark, 0.0)
        target = ambient + contribution * color[None, None, :] * intensity * transition
        return ambient, target, {
            "r": float(color[0]),
            "g": float(color[1]),
            "b": float(color[2]),
            "intensity": intensity,
            "transition": transition,
            "fixture_mask": fixture.get("mask"),
        }

    def _supports(self, scene: dict[str, Any], task: str) -> bool:
        if task == "ambient_scale":
            return bool(scene.get("ambient"))
        if task == "global_diffuse":
            return len(scene.get("diffuse", [])) >= 2
        if task == "add_light":
            return bool(scene.get("point_lights"))
        return bool(self._usable_fixtures(scene))

    def _usable_fixtures(self, scene: dict[str, Any]) -> list[dict[str, Any]]:
        fixtures = [
            item
            for item in self._fixtures(scene)
            if item.get("path") or item.get("on")
        ]
        if self.fixture_mask_enabled:
            fixtures = [item for item in fixtures if item.get("mask")]
        return fixtures

    @staticmethod
    def _fixtures(scene: dict[str, Any]) -> list[dict[str, Any]]:
        return scene.get("in_scene_lights") or scene.get("fixtures") or []

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def _read_linear(self, value: str | Path) -> np.ndarray:
        path = self._resolve(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".npy":
            image = np.load(path).astype(np.float32)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"cannot read linear image: {path}")
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected HWC RGB image, got {path}: {image.shape}")
        image = image[..., :3]
        if not np.isfinite(image).all():
            raise ValueError(f"linear image contains NaN/Inf: {path}")
        return image

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = center_crop(np.maximum(image * self.exposure, 0.0))
        image = image / (1.0 + image)
        if image.shape[:2] != (self.resolution, self.resolution):
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image * 2.0 - 1.0, dtype=np.float32)

    def _read_mask(self, value: str | Path) -> np.ndarray:
        path = self._resolve(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"cannot read fixture mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        original_dtype = mask.dtype
        mask = mask.astype(np.float32)
        if np.issubdtype(original_dtype, np.integer):
            mask /= float(np.iinfo(original_dtype).max)
        elif float(mask.max(initial=0.0)) > 1.0:
            mask /= 255.0
        if not np.isfinite(mask).all():
            raise ValueError(f"fixture mask contains NaN/Inf: {path}")
        return np.clip(mask, 0.0, 1.0)

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = center_crop(mask)
        if mask.shape != (self.resolution, self.resolution):
            mask = cv2.resize(
                mask,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)


def center_crop(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return array[top : top + side, left : left + side]


def load_condition_image(path: str | Path, resolution: int, exposure: float) -> Image.Image:
    """Apply the training display path to EXR, or resize a display-ready RGB image."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".exr":
        with Image.open(path) as image:
            return ImageOps.fit(image.convert("RGB"), (resolution, resolution), Image.Resampling.LANCZOS)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read EXR: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    linear = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    if not np.isfinite(linear).all():
        raise ValueError(f"EXR contains NaN/Inf: {path}")
    linear = np.maximum(center_crop(linear) * float(exposure), 0.0)
    mapped = linear / (1.0 + linear)
    mapped = cv2.resize(mapped, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.clip(mapped * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def load_fixture_mask(path: str | Path, resolution: int) -> torch.Tensor:
    """Load a fixture mask with the same center crop/resize geometry as training."""
    mask_path = Path(path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"cannot read fixture mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    original_dtype = mask.dtype
    mask = mask.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        mask /= float(np.iinfo(original_dtype).max)
    elif float(mask.max(initial=0.0)) > 1.0:
        mask /= 255.0
    mask = center_crop(np.clip(mask, 0.0, 1.0))
    if mask.shape != (resolution, resolution):
        mask = cv2.resize(
            mask,
            (resolution, resolution),
            interpolation=cv2.INTER_AREA,
        )
    mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)
    if not np.isfinite(mask).all() or float(mask.max(initial=0.0)) <= 0.0:
        raise ValueError(f"fixture mask is empty or invalid: {mask_path}")
    return torch.from_numpy(mask)[None]


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "condition_pixel_values": torch.stack([sample["condition_pixel_values"] for sample in samples]),
        "target_pixel_values": torch.stack([sample["target_pixel_values"] for sample in samples]),
        "lighting_values": torch.stack([sample["lighting_values"] for sample in samples]),
        "lighting_known": torch.stack([sample["lighting_known"] for sample in samples]),
        "lighting_valid": torch.stack([sample["lighting_valid"] for sample in samples]),
        "lighting_task_ids": torch.stack([sample["lighting_task_id"] for sample in samples]),
        "fixture_mask": torch.stack([sample["fixture_mask"] for sample in samples]),
        "fixture_present": torch.stack([sample["fixture_present"] for sample in samples]),
        "control": [sample["control"] for sample in samples],
        "task_name": [sample["task_name"] for sample in samples],
        "scene_id": [sample["scene_id"] for sample in samples],
        "sample_index": [sample["sample_index"] for sample in samples],
        "sample_seed": [sample["sample_seed"] for sample in samples],
    }
