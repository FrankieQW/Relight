from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MODEL_ENTRIES = (
    "model_index.json",
    "scheduler",
    "transformer",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    config = deepcopy(loaded)
    config["_config_path"] = str(config_path)
    _validate(config)
    return config


def _require(config: dict[str, Any], dotted_name: str) -> Any:
    value: Any = config
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"missing configuration value: {dotted_name}")
        value = value[part]
    if value is None or value == "":
        raise ConfigError(f"empty configuration value: {dotted_name}")
    return value


def _validate(config: dict[str, Any]) -> None:
    required = (
        "paths.pretrained_model",
        "paths.dataset_root",
        "paths.train_manifest",
        "paths.validation_manifest",
        "paths.output_root",
        "data.resolution",
        "data.exposure",
        "data.samples_per_scene.train",
        "data.samples_per_scene.validation",
        "data.seed",
        "data.tasks",
        "data.task_probabilities",
        "data.ranges.ambient_scale",
        "data.ranges.light_color",
        "data.ranges.light_intensity",
        "data.ranges.fixture_intensity",
        "data.ranges.fixture_transition",
        "model.max_lights",
        "model.fixture_mask_enabled",
        "model.fixture_mask_stride",
        "model.fixture_mask_hidden_dim",
        "lighting.fourier_features",
        "lighting.hidden_dim",
        "lighting.fourier_sigma",
        "lighting.fourier_seed",
        "lora.rank",
        "lora.alpha",
        "lora.dropout",
        "lora.target_modules",
        "train.mixed_precision",
        "train.micro_batch_size",
        "train.gradient_accumulation_steps",
        "train.max_steps",
        "train.learning_rate",
        "train.weight_decay",
        "train.max_grad_norm",
        "train.checkpointing_steps",
        "train.seed",
        "train.run_name",
        "infer.steps",
        "infer.guidance_scale",
        "infer.seed",
    )
    for name in required:
        _require(config, name)

    resolution = int(config["data"]["resolution"])
    if resolution < 256 or resolution % 16:
        raise ConfigError("data.resolution must be >= 256 and divisible by 16")
    allowed_tasks = {"ambient_scale", "global_diffuse", "add_light", "in_scene_light"}
    tasks = list(config["data"]["tasks"])
    unknown = set(tasks) - allowed_tasks
    if unknown:
        raise ConfigError(f"unknown data.tasks: {sorted(unknown)}")
    probabilities = config["data"]["task_probabilities"]
    if any(float(probabilities.get(task, 0.0)) < 0 for task in tasks):
        raise ConfigError("task probabilities cannot be negative")
    if sum(float(probabilities.get(task, 0.0)) for task in tasks) <= 0:
        raise ConfigError("enabled task probabilities must have a positive sum")
    for name, value in config["data"]["ranges"].items():
        if not isinstance(value, list) or len(value) != 2 or float(value[0]) > float(value[1]):
            raise ConfigError(f"data.ranges.{name} must be [minimum, maximum]")
    if int(config["model"]["max_lights"]) < 1:
        raise ConfigError("model.max_lights must be positive")
    fixture_stride = int(config["model"]["fixture_mask_stride"])
    if fixture_stride < 1 or int(config["model"]["fixture_mask_hidden_dim"]) < 1:
        raise ConfigError("fixture mask stride and hidden dimension must be positive")
    packed_resolution = resolution // 16
    if bool(config["model"]["fixture_mask_enabled"]) and packed_resolution % fixture_stride:
        raise ConfigError(
            "data.resolution / 16 must be divisible by model.fixture_mask_stride"
        )
    if int(config["lighting"]["fourier_features"]) < 1 or int(config["lighting"]["hidden_dim"]) < 1:
        raise ConfigError("lighting.fourier_features and lighting.hidden_dim must be positive")
    lighting_attention_mass = float(config["model"].get("lighting_attention_mass", 0.05))
    if not 0.0 < lighting_attention_mass < 1.0:
        raise ConfigError("model.lighting_attention_mass must be between 0 and 1")
    bias_enabled = config["model"].get("lighting_attention_bias_enabled", True)
    if not isinstance(bias_enabled, bool):
        raise ConfigError("model.lighting_attention_bias_enabled must be a boolean")
    if int(config["lora"]["rank"]) < 1 or int(config["lora"]["alpha"]) < 1:
        raise ConfigError("LoRA rank and alpha must be positive")
    if not list(config["lora"]["target_modules"]):
        raise ConfigError("lora.target_modules cannot be empty")
    if config["train"]["mixed_precision"] not in {"bf16", "fp16", "no"}:
        raise ConfigError("train.mixed_precision must be bf16, fp16, or no")
    if config["train"].get("vae_encode_mode", "mode") not in {"mode", "sample"}:
        raise ConfigError("train.vae_encode_mode must be mode or sample")
    for name, default in (("validation_steps", 500), ("validation_batches", 8)):
        if int(config["train"].get(name, default)) < 1:
            raise ConfigError(f"train.{name} must be positive")


def validate_model_snapshot(path_value: str | Path, require_exists: bool = True) -> Path:
    model_path = Path(path_value).expanduser()
    if not model_path.exists():
        if require_exists:
            raise FileNotFoundError(
                f"FLUX.1-Kontext-dev snapshot is missing: {model_path}. "
                "Download the complete Diffusers snapshot to that directory."
            )
        return model_path
    missing = [entry for entry in REQUIRED_MODEL_ENTRIES if not (model_path / entry).exists()]
    if missing:
        raise ConfigError(f"incomplete Diffusers model snapshot at {model_path}; missing: {missing}")
    return model_path
