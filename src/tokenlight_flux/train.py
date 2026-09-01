from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import REQUIRED_MODEL_ENTRIES, load_config, validate_model_snapshot
from .dataset import TokenLightKontextDataset, collate
from .lighting import (
    LightingConditionedTransformer,
    LightingSchema,
    LightingTokenEncoder,
    make_text_free_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FLUX.1-Kontext-dev LoRA on TokenLight pairs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Temporary override, useful for a smoke run")
    parser.add_argument("--resume", help="Override paths.resume_checkpoint; use 'latest' or a checkpoint directory")
    return parser.parse_args()


def dtype_for(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[name]


def encode_vae(
    vae: torch.nn.Module,
    images: torch.Tensor,
    dtype: torch.dtype,
    encode_mode: str,
) -> torch.Tensor:
    device = next(vae.parameters()).device
    posterior = vae.encode(images.to(device=device, dtype=dtype)).latent_dist
    latents = posterior.mode() if encode_mode == "mode" else posterior.sample()
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    scale = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    return (latents - shift) * scale


def pack_latents(pipeline_class: type, latents: torch.Tensor) -> torch.Tensor:
    return pipeline_class._pack_latents(
        latents,
        batch_size=latents.shape[0],
        num_channels_latents=latents.shape[1],
        height=latents.shape[2],
        width=latents.shape[3],
    )


def make_image_ids(
    pipeline_class: type,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    condition: bool,
) -> torch.Tensor:
    ids = pipeline_class._prepare_latent_image_ids(
        latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2, device, dtype
    )
    if condition:
        ids = ids.clone()
        ids[..., 0] = 1
    return ids


def calculate_dynamic_shift(scheduler_config: Any, image_seq_len: int) -> float:
    """Calculate FLUX's resolution-dependent flow shift from scheduler metadata."""
    if image_seq_len < 1:
        raise ValueError(f"image_seq_len must be positive, got {image_seq_len}")
    required = (
        "base_image_seq_len",
        "max_image_seq_len",
        "base_shift",
        "max_shift",
    )
    missing = [name for name in required if getattr(scheduler_config, name, None) is None]
    if missing:
        raise ValueError(
            "dynamic-shifting scheduler config is incomplete; missing " + ", ".join(missing)
        )
    base_seq_len = int(scheduler_config.base_image_seq_len)
    max_seq_len = int(scheduler_config.max_image_seq_len)
    if max_seq_len <= base_seq_len:
        raise ValueError(
            "scheduler max_image_seq_len must exceed base_image_seq_len, got "
            f"{max_seq_len} <= {base_seq_len}"
        )
    base_shift = float(scheduler_config.base_shift)
    max_shift = float(scheduler_config.max_shift)
    slope = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return float(image_seq_len * slope + intercept)


def configure_training_timesteps(
    scheduler: Any,
    num_train_timesteps: int,
    image_seq_len: int,
    device: torch.device,
) -> float | None:
    """Build the training sigma schedule, including FLUX dynamic shifting when enabled."""
    kwargs: dict[str, Any] = {}
    mu: float | None = None
    if bool(getattr(scheduler.config, "use_dynamic_shifting", False)):
        mu = calculate_dynamic_shift(scheduler.config, image_seq_len)
        kwargs["mu"] = mu
    scheduler.set_timesteps(num_train_timesteps, device=device, **kwargs)
    return mu


def resolve_resume(config: dict[str, Any], run_dir: Path) -> Path | None:
    value = config["paths"].get("resume_checkpoint")
    if not value:
        return None
    if str(value) != "latest":
        path = Path(value).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def prune_checkpoints(run_dir: Path, limit: int) -> None:
    if limit < 1:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    for _, path in sorted(checkpoints)[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)


def print_contract(config: dict[str, Any]) -> None:
    model_path = validate_model_snapshot(config["paths"]["pretrained_model"], require_exists=True)
    schema = LightingSchema(int(config["model"]["max_lights"]))
    print(
        json.dumps(
            {
                "config": config["_config_path"],
                "pretrained_model": str(model_path),
                "required_model_entries": list(REQUIRED_MODEL_ENTRIES),
                "dataset_root": config["paths"]["dataset_root"],
                "resolution": int(config["data"]["resolution"]),
                "tasks": list(config["data"]["tasks"]),
                "lora_rank": int(config["lora"]["rank"]),
                "lora_targets": list(config["lora"]["target_modules"]),
                "lighting_token_count": len(schema.names),
                "lighting_schema": list(schema.names),
                "lighting_fourier_features": int(config["lighting"]["fourier_features"]),
                "network_access": "disabled by local_files_only=True",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def inspect_data(config: dict[str, Any]) -> None:
    for split in ("train", "validation"):
        dataset = TokenLightKontextDataset(config, split)
        sample = dataset[0]
        print(
            json.dumps(
                {
                    "split": split,
                    "length": len(dataset),
                    "condition_shape": list(sample["condition_pixel_values"].shape),
                    "target_shape": list(sample["target_pixel_values"].shape),
                    "condition_range": [
                        float(sample["condition_pixel_values"].min()),
                        float(sample["condition_pixel_values"].max()),
                    ],
                    "target_range": [
                        float(sample["target_pixel_values"].min()),
                        float(sample["target_pixel_values"].max()),
                    ],
                    "task": sample["task_name"],
                    "control": sample["control"],
                    "lighting_token_count": int(sample["lighting_values"].numel()),
                    "lighting_known_count": int(sample["lighting_known"].sum()),
                    "lighting_valid_count": int(sample["lighting_valid"].sum()),
                    "scene_id": sample["scene_id"],
                    "sample_seed": sample["sample_seed"],
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["train"]["max_steps"] = args.max_steps
    if args.resume is not None:
        config["paths"]["resume_checkpoint"] = args.resume
    if args.check_config:
        print_contract(config)
        return
    if args.check_data:
        inspect_data(config)
        return

    model_path = validate_model_snapshot(config["paths"]["pretrained_model"])

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxKontextPipeline
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

    train_config = config["train"]
    mixed_precision = str(train_config["mixed_precision"])
    output_root = Path(config["paths"]["output_root"]).expanduser()
    run_dir = output_root / str(train_config["run_name"])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        mixed_precision=None if mixed_precision == "no" else mixed_precision,
        log_with="tensorboard",
        project_dir=str(run_dir),
    )
    set_seed(int(train_config["seed"]), device_specific=True)
    weight_dtype = dtype_for(mixed_precision)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config["_config_path"], run_dir / "config.yaml")
    accelerator.wait_for_everyone()

    pipeline = FluxKontextPipeline.from_pretrained(
        model_path, dtype=weight_dtype, local_files_only=True
    )
    transformer = pipeline.transformer
    vae = pipeline.vae
    for module in (transformer, vae):
        module.requires_grad_(False)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()

    lora_config = config["lora"]
    transformer.add_adapter(
        LoraConfig(
            r=int(lora_config["rank"]),
            lora_alpha=int(lora_config["alpha"]),
            lora_dropout=float(lora_config["dropout"]),
            init_lora_weights="gaussian",
            target_modules=list(lora_config["target_modules"]),
        )
    )
    schema = LightingSchema(int(config["model"]["max_lights"]))
    context_dim = int(transformer.config.joint_attention_dim)
    lighting_config = config["lighting"]
    lighting_encoder = LightingTokenEncoder(
        schema=schema,
        context_dim=context_dim,
        hidden_dim=int(lighting_config["hidden_dim"]),
        fourier_features=int(lighting_config["fourier_features"]),
        fourier_sigma=float(lighting_config["fourier_sigma"]),
        fourier_seed=int(lighting_config["fourier_seed"]),
    )
    model = LightingConditionedTransformer(transformer, lighting_encoder)
    # Text is not a model condition in this method. Drop the pretrained text
    # components after pipeline construction so they are neither encoded nor moved to GPU.
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced zero trainable parameters")
    if mixed_precision == "fp16":
        for parameter in trainable:
            parameter.data = parameter.data.float()
    lora_parameter_count = sum(
        parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad
    )
    lighting_parameter_count = sum(parameter.numel() for parameter in lighting_encoder.parameters())

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_config["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(train_config["weight_decay"]),
        eps=1.0e-8,
    )
    dataset = TokenLightKontextDataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["micro_batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(train_config["seed"])),
    )

    def save_hook(models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], path: str) -> None:
        if accelerator.is_main_process:
            from safetensors.torch import save_file

            unwrapped = accelerator.unwrap_model(models[0])
            state = get_peft_model_state_dict(unwrapped.transformer)
            FluxKontextPipeline.save_lora_weights(
                path, transformer_lora_layers=state, safe_serialization=True
            )
            lighting_state = {
                key: value.detach().cpu().contiguous()
                for key, value in unwrapped.lighting_encoder.state_dict().items()
            }
            save_file(lighting_state, str(Path(path) / "lighting_encoder.safetensors"))
            (Path(path) / "lighting_config.json").write_text(
                json.dumps(
                    {
                        "schema_names": list(schema.names),
                        "max_lights": schema.max_lights,
                        "context_dim": context_dim,
                        "hidden_dim": int(lighting_config["hidden_dim"]),
                        "fourier_features": int(lighting_config["fourier_features"]),
                        "fourier_sigma": float(lighting_config["fourier_sigma"]),
                        "fourier_seed": int(lighting_config["fourier_seed"]),
                        "text_conditioning": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        while weights:
            weights.pop()

    def load_hook(models: list[torch.nn.Module], path: str) -> None:
        from safetensors.torch import load_file

        model = accelerator.unwrap_model(models.pop())
        state = FluxKontextPipeline.lora_state_dict(path, local_files_only=True)
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        incompatible = set_peft_model_state_dict(
            model.transformer,
            transformer_state,
            adapter_name="default",
        )
        if getattr(incompatible, "unexpected_keys", None):
            raise RuntimeError(f"unexpected LoRA keys while resuming: {incompatible.unexpected_keys}")
        lighting_path = Path(path) / "lighting_encoder.safetensors"
        metadata_path = Path(path) / "lighting_config.json"
        if not lighting_path.is_file():
            raise FileNotFoundError(f"checkpoint has no lighting encoder: {lighting_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint has no lighting metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "schema_names": list(schema.names),
            "max_lights": schema.max_lights,
            "context_dim": context_dim,
            "hidden_dim": int(lighting_config["hidden_dim"]),
            "fourier_features": int(lighting_config["fourier_features"]),
            "fourier_sigma": float(lighting_config["fourier_sigma"]),
            "fourier_seed": int(lighting_config["fourier_seed"]),
            "text_conditioning": False,
        }
        if metadata != expected_metadata:
            raise RuntimeError(
                f"lighting checkpoint metadata mismatch: expected={expected_metadata}, actual={metadata}"
            )
        missing, unexpected = model.lighting_encoder.load_state_dict(
            load_file(str(lighting_path), device="cpu"), strict=True
        )
        if missing or unexpected:
            raise RuntimeError(f"lighting checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # Preserve any per-module dtype choices made by Diffusers while moving the VAE.
    vae.to(accelerator.device).eval()
    pipeline.vae = vae

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    scheduler_image_seq_len: int | None = None
    scheduler_mu: float | None = None

    global_step = 0
    resume_path = resolve_resume(config, run_dir)
    if resume_path is not None:
        accelerator.load_state(str(resume_path))
        global_step = int(resume_path.name.rsplit("-", 1)[1])

    max_steps = int(train_config["max_steps"])
    checkpoint_every = int(train_config["checkpointing_steps"])
    max_grad_norm = float(train_config["max_grad_norm"])
    guidance_scale = float(train_config.get("guidance_scale", 3.5))
    vae_encode_mode = str(train_config.get("vae_encode_mode", "mode"))
    accelerator.init_trackers(str(train_config["run_name"]))
    model.train()

    while global_step < max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    target_latents = encode_vae(
                        vae, batch["target_pixel_values"], weight_dtype, vae_encode_mode
                    )
                    condition_latents = encode_vae(
                        vae, batch["condition_pixel_values"], weight_dtype, vae_encode_mode
                    )
                    context_embeds, pooled_context, context_ids = make_text_free_condition(
                        accelerator.unwrap_model(model).transformer,
                        batch_size=batch["target_pixel_values"].shape[0],
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                target_ids = make_image_ids(
                    FluxKontextPipeline, target_latents, accelerator.device, weight_dtype, False
                )
                condition_ids = make_image_ids(
                    FluxKontextPipeline, condition_latents, accelerator.device, weight_dtype, True
                )
                clean_target = pack_latents(FluxKontextPipeline, target_latents)
                condition = pack_latents(FluxKontextPipeline, condition_latents)
                current_image_seq_len = int(clean_target.shape[1])
                if scheduler_image_seq_len != current_image_seq_len:
                    scheduler_mu = configure_training_timesteps(
                        scheduler,
                        num_train_timesteps=num_train_timesteps,
                        image_seq_len=current_image_seq_len,
                        device=accelerator.device,
                    )
                    scheduler_image_seq_len = current_image_seq_len
                    if accelerator.is_main_process:
                        print(
                            json.dumps(
                                {
                                    "scheduler": "FlowMatchEulerDiscreteScheduler",
                                    "resolution": int(config["data"]["resolution"]),
                                    "target_image_seq_len": scheduler_image_seq_len,
                                    "dynamic_shift_mu": scheduler_mu,
                                    "num_train_timesteps": num_train_timesteps,
                                }
                            ),
                            flush=True,
                        )
                noise = torch.randn_like(clean_target)
                timestep_indices = torch.randint(
                    0, num_train_timesteps, (clean_target.shape[0],), device=accelerator.device
                )
                timesteps = scheduler.timesteps[timestep_indices].to(dtype=weight_dtype)
                sigmas = scheduler.sigmas[timestep_indices].to(
                    device=accelerator.device, dtype=clean_target.dtype
                ).view(-1, *([1] * (clean_target.ndim - 1)))
                noisy_target = (1.0 - sigmas) * clean_target + sigmas * noise
                hidden_states = torch.cat((noisy_target, condition), dim=1)
                image_ids = torch.cat((target_ids, condition_ids), dim=-2)

                raw_transformer = accelerator.unwrap_model(model).transformer
                guidance = None
                if bool(getattr(raw_transformer.config, "guidance_embeds", False)):
                    guidance = torch.full(
                        (hidden_states.shape[0],),
                        guidance_scale,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                autocast = (
                    torch.autocast("cuda", dtype=weight_dtype)
                    if accelerator.device.type == "cuda" and mixed_precision != "no"
                    else nullcontext()
                )
                with autocast:
                    prediction = model(
                        hidden_states=hidden_states,
                        timestep=timesteps / 1000.0,
                        guidance=guidance,
                        pooled_projections=pooled_context,
                        encoder_hidden_states=context_embeds,
                        txt_ids=context_ids,
                        img_ids=image_ids,
                        lighting_values=batch["lighting_values"],
                        lighting_known=batch["lighting_known"],
                        lighting_valid=batch["lighting_valid"],
                        lighting_task_ids=batch["lighting_task_ids"],
                        return_dict=False,
                    )[0]
                    prediction = prediction[:, : clean_target.shape[1]]
                    flow_target = noise - clean_target
                    loss = torch.mean((prediction.float() - flow_target.float()) ** 2)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                mean_loss = accelerator.gather(loss.detach().repeat(hidden_states.shape[0])).mean().item()
                accelerator.log({"train/loss": mean_loss}, step=global_step)
                if accelerator.is_main_process and global_step % 10 == 0:
                    print(f"step={global_step}/{max_steps} loss={mean_loss:.6f}", flush=True)
                if global_step % checkpoint_every == 0:
                    accelerator.save_state(str(run_dir / f"checkpoint-{global_step}"))
                    if accelerator.is_main_process:
                        prune_checkpoints(run_dir, int(train_config.get("checkpoints_total_limit", 0)))
                if global_step >= max_steps:
                    break

    accelerator.wait_for_everyone()
    final_dir = run_dir / "final"
    accelerator.save_state(str(final_dir))
    if accelerator.is_main_process:
        (final_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "base_model": str(model_path),
                    "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                    "lora_parameters": lora_parameter_count,
                    "lighting_encoder_parameters": lighting_parameter_count,
                    "world_size": accelerator.num_processes,
                    "target_image_seq_len": scheduler_image_seq_len,
                    "dynamic_shift_mu": scheduler_mu,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.end_training()
