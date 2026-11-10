from __future__ import annotations

import hashlib
import json
import os
import uuid
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

    def __init__(self, config: dict[str, Any], split: str, *, materialize: bool = False):
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError(f"unsupported split: {split}")
        self.config = config
        self.split = split
        # Composed PNG/metadata export is an explicit offline pass
        # (tokenlight_flux.precompose). Training leaves this off so that
        # sampling never writes to disk.
        self.materialize = bool(materialize)
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
        intrinsic_data = data.get("intrinsics") or {}
        if not isinstance(intrinsic_data, dict):
            raise ValueError("data.intrinsics must be a mapping")
        self.intrinsics_enabled = bool(intrinsic_data.get("enabled", False))
        self.intrinsics_require = bool(intrinsic_data.get("require", False))
        self.head_resolution = int(
            (config.get("intrinsic") or {}).get("head_resolution", self.resolution // 8)
        )
        self.schema = LightingSchema(self.max_lights)
        self.ambient_range = _range(config, "ambient_scale")
        self.color_range = _range(config, "light_color")
        self.intensity_range = _range(config, "light_intensity")
        self.fixture_intensity_range = _range(config, "fixture_intensity")
        self.fixture_transition_range = _range(config, "fixture_transition")
        composed_root = config["paths"].get("composed_output_root")
        # Separate exports when sampling settings or the manifest change.
        provenance = {
            "format_version": 1,
            "data": data,
            "model": config["model"],
            "dataset_root": str(self.root.resolve()),
            "scenes": self.scenes,
        }
        self.export_id = hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        self.composed_output_root = (
            Path(composed_root).expanduser() / self.export_id / split
            if composed_root and self.materialize
            else None
        )
        if self.composed_output_root is not None:
            self.composed_output_root.mkdir(parents=True, exist_ok=True)

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
        intrinsic: dict[str, np.ndarray] = {}
        has_intrinsic = False
        if self.intrinsics_enabled:
            scene_intr = self._scene_intrinsics(scene)
            if scene_intr:
                has_intrinsic = True
                irradiance = self._compose_irradiance(scene, task, control, scene_intr)
                intrinsic = {
                    "intrinsic_albedo": self._prepare_intrinsic(scene_intr["albedo"], "albedo"),
                    "intrinsic_normal": self._prepare_intrinsic(scene_intr["normal"], "normal"),
                    "intrinsic_irradiance": self._prepare_intrinsic(irradiance, "irradiance"),
                    "intrinsic_mask": self._prepare_intrinsic_mask(scene_intr["valid"]),
                }
            else:
                # data.intrinsics.require=false: keep the batch contract
                # homogeneous but mask every pixel, so this sample adds nothing
                # to either the AIN numerator or its denominator.
                size = self.head_resolution
                intrinsic = {
                    "intrinsic_albedo": np.zeros((size, size, 3), dtype=np.float32),
                    "intrinsic_normal": np.zeros((size, size, 3), dtype=np.float32),
                    "intrinsic_irradiance": np.zeros((size, size, 3), dtype=np.float32),
                    "intrinsic_mask": np.zeros((size, size), dtype=np.float32),
                }
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
        condition_pixels = self._prepare_image(source)
        target_pixels = self._prepare_image(target)
        intrinsic_pixels = {}
        if has_intrinsic:
            # Display mapping: _to_uint8 expects [-1, 1] for every channel.
            intrinsic_pixels = {
                "albedo": intrinsic["intrinsic_albedo"] * 2.0 - 1.0,
                "normal": intrinsic["intrinsic_normal"],
                "irradiance": intrinsic["intrinsic_irradiance"] * 2.0 - 1.0,
            }
        if self.materialize and self.composed_output_root is not None:
            self._save_composed_sample(
                index=index,
                sample_seed=sample_seed,
                scene_id=str(scene["id"]),
                task=task,
                control=control,
                lighting_values=lighting.values,
                lighting_known=lighting.known,
                lighting_valid=lighting.valid,
                condition_pixels=condition_pixels,
                target_pixels=target_pixels,
                intrinsic_pixels=intrinsic_pixels,
            )
        return {
            "condition_pixel_values": torch.from_numpy(condition_pixels).permute(2, 0, 1),
            "target_pixel_values": torch.from_numpy(target_pixels).permute(2, 0, 1),
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
            **{
                key: (
                    torch.from_numpy(value).permute(2, 0, 1)
                    if value.ndim == 3
                    else torch.from_numpy(value)[None]
                )
                for key, value in intrinsic.items()
            },
        }

    def sample_path(self, index: int) -> Path | None:
        """Directory of one composed sample, or None when export is disabled."""
        if self.composed_output_root is None:
            return None
        return self.composed_output_root / f"sample_{int(index):08d}"

    @staticmethod
    def sample_is_complete(sample_dir: Path) -> bool:
        """A composed sample is usable once metadata and both PNGs exist."""
        return (
            (sample_dir / "metadata.json").is_file()
            and (sample_dir / "condition.png").is_file()
            and (sample_dir / "target.png").is_file()
        )

    def _save_composed_sample(
        self,
        *,
        index: int,
        sample_seed: int,
        scene_id: str,
        task: str,
        control: dict[str, Any],
        lighting_values: np.ndarray,
        lighting_known: np.ndarray,
        lighting_valid: np.ndarray,
        condition_pixels: np.ndarray,
        target_pixels: np.ndarray,
        intrinsic_pixels: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Persist viewable source/target images and exact conditioning metadata."""
        sample_dir = self.sample_path(index)
        if sample_dir is None:
            return
        if self.sample_is_complete(sample_dir):
            return
        metadata_path = sample_dir / "metadata.json"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for name, pixels in (("condition", condition_pixels), ("target", target_pixels)):
            temporary = sample_dir / f".{name}.{uuid.uuid4().hex}.png"
            try:
                Image.fromarray(self._to_uint8(pixels), mode="RGB").save(temporary)
                os.replace(temporary, sample_dir / f"{name}.png")
            finally:
                temporary.unlink(missing_ok=True)
        # AIN passes are exported only for human inspection; they are not
        # training inputs and are absent when intrinsic supervision is off.
        for name, pixels in (intrinsic_pixels or {}).items():
            temporary = sample_dir / f".intrinsic_{name}.{uuid.uuid4().hex}.png"
            try:
                Image.fromarray(self._to_uint8(pixels), mode="RGB").save(temporary)
                os.replace(temporary, sample_dir / f"intrinsic_{name}.png")
            finally:
                temporary.unlink(missing_ok=True)
        metadata = {
            "split": self.split,
            "sample_index": int(index),
            "sample_seed": int(sample_seed),
            "scene_id": scene_id,
            "task": task,
            "control": control,
            "lighting_values": [float(value) for value in lighting_values],
            "lighting_known": [float(value) for value in lighting_known],
            "lighting_valid": [float(value) for value in lighting_valid],
            "resolution": int(self.resolution),
            "exposure": float(self.exposure),
            "source_image": "condition.png",
            "target_image": "target.png",
            "lighting_schema": list(self.schema.names),
        }
        temporary_metadata = sample_dir / f".metadata.{uuid.uuid4().hex}.json"
        try:
            temporary_metadata.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary_metadata.unlink(missing_ok=True)
        if self.composed_output_root is not None:
            manifest_path = self.composed_output_root / "export_config.json"
            if not manifest_path.exists():
                manifest_path.write_text(
                    json.dumps(
                        {
                            "export_id": self.export_id,
                            "split": self.split,
                            "sample_count": len(self),
                            "ambient_scale_range": list(self.ambient_range),
                            "config_output_root": str(self.composed_output_root),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.clip((image + 1.0) * 127.5 + 0.5, 0, 255), dtype=np.uint8)

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
                # AIN irradiance must be rebuilt from the components that actually
                # took part in this composition.
                "source_index": int(source_index),
                "target_index": int(target_index),
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
                position = component["camera_position"]
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
                        "component_index": int(component_index),
                    }
                )
            return ambient, target, {"lights": lights}

        fixtures = self._usable_fixtures(scene)
        selection = int(rng.integers(len(fixtures)))
        fixture = fixtures[selection]
        # Render passes are named by a fixture's position in the scene's own list,
        # not by its position after mask/usability filtering.
        fixture_index = next(
            (index for index, item in enumerate(self._fixtures(scene)) if item is fixture),
            selection,
        )
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
            # Index into scene["in_scene_lights"], matching passes/irradiance naming.
            "fixture_index": fixture_index,
            "fixture_component": fixture,
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

    def _read_map(self, value: str | Path) -> np.ndarray:
        """Read EXR/NPY values without exposure or tone mapping."""
        path = self._resolve(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".npy":
            image = np.load(path).astype(np.float32)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"cannot read intrinsic map: {path}")
            image = image.astype(np.float32)
            if image.ndim == 3:
                image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected HWC RGB intrinsic map, got {path}: {image.shape}")
        image = image[..., :3]
        if not np.isfinite(image).all():
            raise ValueError(f"intrinsic map contains NaN/Inf: {path}")
        return image

    @staticmethod
    def _component_path(component: dict[str, Any]) -> str | Path | None:
        """A light component renders to whatever field the dataset exposes."""
        return component.get("path") or component.get("on")

    def _scene_intrinsics(self, scene: dict[str, Any]) -> dict[str, Any]:
        """Locate one scene's AIN passes.

        Passes live in ``<component>/passes/``. The metadata ``passes`` block is
        honoured when present, but the directory layout is authoritative on its
        own, so a manifest line without a ``passes`` block still resolves.
        """
        passes = scene.get("passes") or {}
        if not isinstance(passes, dict):
            raise ValueError(f"scene passes block must be a mapping: {scene.get('id')}")
        if not scene.get("ambient"):
            raise ValueError(f"scene has no ambient render to locate passes: {scene.get('id')}")
        passes_dir = self._resolve(scene["ambient"]).parent / "passes"
        albedo = passes.get("albedo") or (passes_dir / "albedo.exr")
        normal = passes.get("normal_camera") or (passes_dir / "normal_camera.exr")
        mask = passes.get("valid_mask") or (passes_dir / "valid_mask.png")
        if not all(self._resolve(path).is_file() for path in (albedo, normal, mask)):
            if self.intrinsics_require:
                raise ValueError(
                    "scene has no AIN passes (expected albedo/normal_camera/valid_mask in "
                    f"{passes_dir}): {scene.get('id')}"
                )
            return {}
        return {
            "albedo": self._read_map(albedo),
            "normal": self._read_map(normal),
            "valid": self._read_mask(mask),
            "irradiance": dict(passes.get("irradiance") or {}),
            "irradiance_dir": passes_dir / "irradiance",
        }

    def _compose_irradiance(
        self, scene: dict[str, Any], task: str, control: dict[str, Any], intr: dict[str, Any]
    ) -> np.ndarray:
        """Rebuild E for the target composition so the AIN target matches the image."""
        # Division by zero on background pixels would poison the map; the valid
        # mask keeps those pixels out of the loss anyway.
        albedo = np.maximum(intr["albedo"], 1e-3)
        dark = self._read_linear(scene["dark"]) if scene.get("dark") else 0.0
        index = intr["irradiance"]
        irradiance_dir = intr.get("irradiance_dir")

        def E_of(component: dict[str, Any], name: str) -> np.ndarray:
            # 1) explicit index from the metadata passes block
            path = index.get(name)
            # 2) the renderer writes <component>/passes/irradiance/<name>.exr
            if not path and irradiance_dir is not None:
                for suffix in (".exr", ".npy"):
                    candidate = irradiance_dir / f"{name}{suffix}"
                    if candidate.is_file():
                        path = candidate
                        break
            if path:
                return self._read_map(path)
            # 3) algebraically undo the render: E = (radiance - dark) / albedo
            value = self._component_path(component)
            if not value:
                raise ValueError(f"light component has no render path: {name}")
            radiance = self._read_linear(value)
            return np.maximum(radiance - dark, 0.0) / albedo

        e_ambient = E_of({"path": scene["ambient"]}, "ambient")

        if task == "ambient_scale":
            return e_ambient * float(control["scale"])

        if task == "global_diffuse":
            target_index = int(control["target_index"])
            component = scene["diffuse"][target_index]
            return e_ambient + E_of(component, f"diffuse_{target_index:02d}")

        if task == "add_light":
            total = e_ambient.copy()
            for light in control["lights"]:
                component_index = int(light["component_index"])
                component = scene["point_lights"][component_index]
                rgb = np.asarray([light["r"], light["g"], light["b"]], dtype=np.float32)
                total = total + E_of(component, f"light_{component_index:03d}") * rgb[
                    None, None, :
                ] * float(light["intensity"])
            return total

        component = control["fixture_component"]
        rgb = np.asarray([control["r"], control["g"], control["b"]], dtype=np.float32)
        return e_ambient + E_of(
            component, f"fixture_{int(control['fixture_index']):03d}"
        ) * rgb[None, None, :] * float(control["intensity"]) * float(control["transition"])

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = center_crop(np.maximum(image * self.exposure, 0.0))
        image = image / (1.0 + image)
        if image.shape[:2] != (self.resolution, self.resolution):
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image * 2.0 - 1.0, dtype=np.float32)

    def _prepare_intrinsic(self, image: np.ndarray, kind: str) -> np.ndarray:
        """Center-crop/resize an AIN map to the supervision grid and normalise it."""
        value = center_crop(image)
        if value.shape[:2] != (self.head_resolution, self.head_resolution):
            value = cv2.resize(
                value,
                (self.head_resolution, self.head_resolution),
                interpolation=cv2.INTER_AREA,
            )
        if kind == "albedo":
            value = np.clip(value, 0.0, 1.0)
        elif kind == "normal":
            norm = np.linalg.norm(value, axis=2, keepdims=True)
            value = np.where(norm > 1e-6, value / np.maximum(norm, 1e-6), 0.0)
        elif kind == "irradiance":
            # Same exposure + Reinhard path as the supervised image, so the
            # sigmoid head predicts exactly what the image pipeline represents.
            value = np.maximum(value * self.exposure, 0.0)
            value = value / (1.0 + value)
        else:
            raise ValueError(f"unknown intrinsic kind: {kind}")
        return np.ascontiguousarray(value, dtype=np.float32)

    def _prepare_intrinsic_mask(self, mask: np.ndarray) -> np.ndarray:
        value = center_crop(mask)
        if value.shape[:2] != (self.head_resolution, self.head_resolution):
            value = cv2.resize(
                value,
                (self.head_resolution, self.head_resolution),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray((value > 0.5).astype(np.float32), dtype=np.float32)

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
    batch = {
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
    # Only present when data.intrinsics.enabled; keeping the keys out entirely
    # is what makes the disabled path identical to the plain flow-matching run.
    for key in ("intrinsic_albedo", "intrinsic_normal", "intrinsic_irradiance", "intrinsic_mask"):
        if all(key in sample for sample in samples):
            batch[key] = torch.stack([sample[key] for sample in samples])
    return batch
