from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

import torch

from .config import load_config, validate_model_snapshot
from .dataset import load_condition_image, load_fixture_mask
from .lighting import (
    FixtureMaskEncoder,
    LightingConditionedTransformer,
    LightingSchema,
    LightingTokenEncoder,
    make_text_free_condition,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TokenLight FLUX-Kontext LoRA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora", required=True, help="checkpoint-N or final directory")
    parser.add_argument("--source", required=True, help="linear EXR or display-ready RGB image")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--task",
        choices=("ambient_scale", "global_diffuse", "add_light", "in_scene_light"),
        default="ambient_scale",
    )
    parser.add_argument("--ambient-scale", type=float)
    parser.add_argument("--source-diffuse", type=float)
    parser.add_argument("--target-diffuse", type=float)
    parser.add_argument(
        "--light",
        action="append",
        help="Repeatable x,y,z,r,g,b,intensity,softness specification for add_light",
    )
    parser.add_argument("--fixture-rgb", help="r,g,b for in_scene_light")
    parser.add_argument("--fixture-intensity", type=float)
    parser.add_argument("--fixture-transition", type=float)
    parser.add_argument("--fixture-mask", help="fixture mask PNG for in_scene_light")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args(argv)


def comma_floats(value: str, count: int, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated numbers") from error
    if len(parsed) != count:
        raise ValueError(f"{name} requires {count} values, got {len(parsed)}")
    return parsed


def structured_control(args: argparse.Namespace) -> dict[str, object]:
    if args.task == "ambient_scale":
        if args.ambient_scale is None:
            raise ValueError("ambient_scale requires --ambient-scale")
        control: dict[str, object] = {"scale": float(args.ambient_scale)}
    elif args.task == "global_diffuse":
        if args.source_diffuse is None or args.target_diffuse is None:
            raise ValueError("global_diffuse requires --source-diffuse and --target-diffuse")
        control = {
            "source_level": float(args.source_diffuse),
            "target_level": float(args.target_diffuse),
            "delta": float(args.target_diffuse - args.source_diffuse),
        }
    elif args.task == "add_light":
        if not args.light:
            raise ValueError("add_light requires at least one --light")
        lights = []
        for value in args.light:
            x, y, z, r, g, b, intensity, softness = comma_floats(value, 8, "--light")
            lights.append(
                {
                    "x": x, "y": y, "z": z, "r": r, "g": g, "b": b,
                    "intensity": intensity, "softness": softness,
                }
            )
        control = {"lights": lights}
    else:
        if args.fixture_rgb is None or args.fixture_intensity is None or args.fixture_transition is None:
            raise ValueError(
                "in_scene_light requires --fixture-rgb, --fixture-intensity, and --fixture-transition"
            )
        r, g, b = comma_floats(args.fixture_rgb, 3, "--fixture-rgb")
        control = {
            "r": r,
            "g": g,
            "b": b,
            "intensity": float(args.fixture_intensity),
            "transition": float(args.fixture_transition),
        }
    return control


def load_tokenlight_runtime(
    config: dict[str, object],
    lora_path: str | Path,
    *,
    cpu_offload: bool = False,
):
    """Load the frozen FLUX base, LoRA, and Lighting Encoder once for inference."""
    from diffusers import FluxKontextPipeline
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; FLUX inference requires a CUDA GPU")
    model_path = validate_model_snapshot(config["paths"]["pretrained_model"])
    lora_path = Path(lora_path).expanduser()
    if not lora_path.is_dir():
        raise FileNotFoundError(lora_path)
    dtype = torch.bfloat16
    pipeline = FluxKontextPipeline.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    )
    pipeline.load_lora_weights(lora_path, local_files_only=True)
    lighting_state_path = lora_path / "lighting_encoder.safetensors"
    lighting_metadata_path = lora_path / "lighting_config.json"
    if not lighting_state_path.is_file() or not lighting_metadata_path.is_file():
        raise FileNotFoundError(
            "LoRA directory must contain lighting_encoder.safetensors and "
            f"lighting_config.json: {lora_path}"
        )
    lighting_metadata = json.loads(lighting_metadata_path.read_text(encoding="utf-8"))
    if lighting_metadata.get("text_conditioning") is not False:
        raise RuntimeError("checkpoint is not marked as text-free TokenLight conditioning")
    schema = LightingSchema(int(config["model"]["max_lights"]))
    if list(schema.names) != list(lighting_metadata["schema_names"]):
        raise RuntimeError("lighting schema in checkpoint does not match the current configuration")
    context_dim = int(pipeline.transformer.config.joint_attention_dim)
    if context_dim != int(lighting_metadata["context_dim"]):
        raise RuntimeError("lighting checkpoint context dimension does not match FLUX")
    lighting_encoder = LightingTokenEncoder(
        schema=schema,
        context_dim=context_dim,
        hidden_dim=int(lighting_metadata["hidden_dim"]),
        fourier_features=int(lighting_metadata["fourier_features"]),
        fourier_sigma=float(lighting_metadata["fourier_sigma"]),
        fourier_seed=int(lighting_metadata["fourier_seed"]),
    )
    lighting_encoder.load_state_dict(
        load_file(str(lighting_state_path), device="cpu"), strict=True
    )
    fixture_mask_encoder = None
    if bool(lighting_metadata.get("fixture_mask_enabled", False)):
        fixture_state_path = lora_path / "fixture_mask_encoder.safetensors"
        if not fixture_state_path.is_file():
            raise FileNotFoundError(
                f"checkpoint metadata requires fixture mask encoder: {fixture_state_path}"
            )
        fixture_mask_encoder = FixtureMaskEncoder(
            context_dim=context_dim,
            hidden_dim=int(lighting_metadata["fixture_mask_hidden_dim"]),
            stride=int(lighting_metadata["fixture_mask_stride"]),
        )
        fixture_mask_encoder.load_state_dict(
            load_file(str(fixture_state_path), device="cpu"), strict=True
        )
    pipeline.transformer = LightingConditionedTransformer(
        pipeline.transformer, lighting_encoder, fixture_mask_encoder
    )
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    return pipeline, schema, dtype, model_path, lora_path


def generate_tokenlight_image(
    pipeline,
    schema: LightingSchema,
    dtype: torch.dtype,
    condition,
    *,
    task: str,
    control: dict[str, object],
    resolution: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    cpu_offload: bool = False,
    fixture_mask: torch.Tensor | None = None,
    fixture_present: bool = False,
):
    """Generate one controlled image from an already-loaded runtime."""
    generator_device = "cpu" if cpu_offload else "cuda"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    packed = schema.pack(task, control)
    lighting_kwargs = {
        "tokenlight_values": torch.from_numpy(packed.values)[None],
        "tokenlight_known": torch.from_numpy(packed.known)[None],
        "tokenlight_valid": torch.from_numpy(packed.valid)[None],
        "tokenlight_task_ids": torch.tensor([packed.task_id], dtype=torch.long),
    }
    if fixture_mask is None:
        fixture_mask = torch.zeros(1, int(resolution), int(resolution), dtype=torch.float32)
    expected_mask_shape = (1, int(resolution), int(resolution))
    if tuple(fixture_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"fixture_mask must have shape {expected_mask_shape}, got "
            f"{tuple(fixture_mask.shape)}"
        )
    lighting_kwargs["tokenlight_fixture_mask"] = fixture_mask[None]
    lighting_kwargs["tokenlight_fixture_present"] = torch.tensor(
        [bool(fixture_present)], dtype=torch.bool
    )
    context_embeds, pooled_context, _ = make_text_free_condition(
        pipeline.transformer.transformer,
        batch_size=1,
        device=pipeline._execution_device,
        dtype=dtype,
    )
    return pipeline(
        image=condition,
        prompt_embeds=context_embeds,
        pooled_prompt_embeds=pooled_context,
        width=int(resolution),
        height=int(resolution),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        joint_attention_kwargs=lighting_kwargs,
    ).images[0]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    lora_path = Path(args.lora).expanduser()
    source_path = Path(args.source).expanduser()
    control = structured_control(args)
    if args.task == "add_light" and len(control.get("lights", [])) > int(config["model"]["max_lights"]):
        raise ValueError(f"add_light accepts at most {config['model']['max_lights']} lights")
    resolution = int(config["data"]["resolution"])
    exposure = float(config["data"]["exposure"])
    condition = load_condition_image(source_path, resolution, exposure)
    seed = int(args.seed if args.seed is not None else config["infer"]["seed"])
    steps = int(args.steps if args.steps is not None else config["infer"]["steps"])
    guidance = float(
        args.guidance_scale if args.guidance_scale is not None else config["infer"]["guidance_scale"]
    )
    fixture_mask = None
    fixture_present = False
    if args.task == "in_scene_light":
        if not args.fixture_mask:
            raise ValueError("in_scene_light requires --fixture-mask")
        fixture_mask = load_fixture_mask(args.fixture_mask, resolution)
        fixture_present = True
    elif args.fixture_mask:
        raise ValueError("--fixture-mask is only valid for in_scene_light")
    pipeline, schema, dtype, model_path, lora_path = load_tokenlight_runtime(
        config, lora_path, cpu_offload=args.cpu_offload
    )
    result = generate_tokenlight_image(
        pipeline,
        schema,
        dtype,
        condition,
        task=args.task,
        control=control,
        resolution=resolution,
        steps=steps,
        guidance_scale=guidance,
        seed=seed,
        cpu_offload=args.cpu_offload,
        fixture_mask=fixture_mask,
        fixture_present=fixture_present,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    metadata = {
        "base_model": str(model_path),
        "lora": str(lora_path),
        "source": str(source_path),
        "output": str(output),
        "task": args.task,
        "control": control,
        "fixture_mask": str(Path(args.fixture_mask).expanduser()) if args.fixture_mask else None,
        "lighting_schema": list(schema.names),
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance,
        "resolution": resolution,
        "exposure": exposure,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))
