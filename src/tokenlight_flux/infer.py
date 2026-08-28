from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config, validate_model_snapshot
from .dataset import load_condition_image
from .lighting import (
    LightingConditionedTransformer,
    LightingSchema,
    LightingTokenEncoder,
    make_text_free_condition,
)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = validate_model_snapshot(config["paths"]["pretrained_model"])
    lora_path = Path(args.lora).expanduser()
    if not lora_path.is_dir():
        raise FileNotFoundError(lora_path)
    source_path = Path(args.source).expanduser()
    control = structured_control(args)
    if args.task == "add_light" and len(control.get("lights", [])) > int(config["model"]["max_lights"]):
        raise ValueError(f"add_light accepts at most {config['model']['max_lights']} lights")

    from diffusers import FluxKontextPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; FLUX inference requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = FluxKontextPipeline.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True
    )
    pipeline.load_lora_weights(lora_path, local_files_only=True)
    lighting_state_path = lora_path / "lighting_encoder.safetensors"
    lighting_metadata_path = lora_path / "lighting_config.json"
    if not lighting_state_path.is_file() or not lighting_metadata_path.is_file():
        raise FileNotFoundError(
            f"LoRA directory must contain lighting_encoder.safetensors and lighting_config.json: {lora_path}"
        )
    lighting_metadata = json.loads(lighting_metadata_path.read_text(encoding="utf-8"))
    if lighting_metadata.get("text_conditioning") is not False:
        raise RuntimeError("checkpoint is not marked as text-free TokenLight conditioning")
    schema = LightingSchema(int(config["model"]["max_lights"]))
    if list(schema.names) != list(lighting_metadata["schema_names"]):
        raise RuntimeError("lighting schema in checkpoint does not match the current configuration")
    context_dim = int(pipeline.transformer.config.joint_attention_dim)
    if context_dim != int(lighting_metadata["context_dim"]):
        raise RuntimeError("lighting checkpoint context dimension does not match the FLUX transformer")
    lighting_encoder = LightingTokenEncoder(
        schema=schema,
        context_dim=context_dim,
        hidden_dim=int(lighting_metadata["hidden_dim"]),
        fourier_features=int(lighting_metadata["fourier_features"]),
        fourier_sigma=float(lighting_metadata["fourier_sigma"]),
        fourier_seed=int(lighting_metadata["fourier_seed"]),
    )
    from safetensors.torch import load_file

    lighting_encoder.load_state_dict(load_file(str(lighting_state_path), device="cpu"), strict=True)
    pipeline.transformer = LightingConditionedTransformer(pipeline.transformer, lighting_encoder)
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")

    resolution = int(config["data"]["resolution"])
    exposure = float(config["data"]["exposure"])
    condition = load_condition_image(source_path, resolution, exposure)
    seed = int(args.seed if args.seed is not None else config["infer"]["seed"])
    steps = int(args.steps if args.steps is not None else config["infer"]["steps"])
    guidance = float(
        args.guidance_scale if args.guidance_scale is not None else config["infer"]["guidance_scale"]
    )
    generator_device = "cpu" if args.cpu_offload else "cuda"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    packed = schema.pack(args.task, control)
    lighting_kwargs = {
        "tokenlight_values": torch.from_numpy(packed.values)[None],
        "tokenlight_known": torch.from_numpy(packed.known)[None],
        "tokenlight_valid": torch.from_numpy(packed.valid)[None],
        "tokenlight_task_ids": torch.tensor([packed.task_id], dtype=torch.long),
    }
    context_embeds, pooled_context, _ = make_text_free_condition(
        pipeline.transformer.transformer,
        batch_size=1,
        device=pipeline._execution_device,
        dtype=dtype,
    )
    result = pipeline(
        image=condition,
        prompt_embeds=context_embeds,
        pooled_prompt_embeds=pooled_context,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        joint_attention_kwargs=lighting_kwargs,
    ).images[0]

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
