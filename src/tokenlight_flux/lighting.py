from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn


TASK_NAMES = ("ambient_scale", "global_diffuse", "add_light", "in_scene_light")
TASK_IDS = {name: index for index, name in enumerate(TASK_NAMES)}
ADD_LIGHT_FIELDS = ("x", "y", "z", "r", "g", "b", "intensity", "softness")
FIXTURE_FIELDS = ("r", "g", "b", "intensity", "transition")


@dataclass(frozen=True)
class PackedLighting:
    values: np.ndarray
    known: np.ndarray
    valid: np.ndarray
    task_id: int


class LightingSchema:
    """Stable scalar-token ordering shared by data, training, checkpoints and inference."""

    def __init__(self, max_lights: int):
        if max_lights < 1:
            raise ValueError("max_lights must be positive")
        names = ["ambient", "global_diffuse"]
        for slot in range(max_lights):
            names.append(f"add_light.{slot}.valid")
            names.extend(f"add_light.{slot}.{field}" for field in ADD_LIGHT_FIELDS)
        names.extend(f"in_scene.{field}" for field in FIXTURE_FIELDS)
        self.max_lights = int(max_lights)
        self.names = tuple(names)
        self.index = {name: index for index, name in enumerate(self.names)}

    def pack(self, task: str, control: dict[str, Any]) -> PackedLighting:
        if task not in TASK_IDS:
            raise ValueError(f"unknown lighting task: {task}")
        count = len(self.names)
        values = np.zeros(count, dtype=np.float32)
        known = np.zeros(count, dtype=np.float32)
        valid = np.zeros(count, dtype=np.float32)

        def set_value(name: str, value: float) -> None:
            index = self.index[name]
            values[index] = np.float32(value)
            known[index] = 1.0
            valid[index] = 1.0

        if task == "ambient_scale":
            set_value("ambient", float(control["scale"]))
        elif task == "global_diffuse":
            set_value("global_diffuse", float(control["delta"]))
        elif task == "add_light":
            lights = list(control["lights"])
            if len(lights) > self.max_lights:
                raise ValueError(f"received {len(lights)} lights, maximum is {self.max_lights}")
            for slot in range(self.max_lights):
                set_value(f"add_light.{slot}.valid", float(slot < len(lights)))
                if slot >= len(lights):
                    continue
                for field in ADD_LIGHT_FIELDS:
                    set_value(f"add_light.{slot}.{field}", float(lights[slot][field]))
        else:
            for field in FIXTURE_FIELDS:
                set_value(f"in_scene.{field}", float(control[field]))
        return PackedLighting(values, known, valid, TASK_IDS[task])


def make_text_free_condition(
    transformer: nn.Module,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the empty context required when lighting tokens are the only context tokens."""
    context_dim = int(transformer.config.joint_attention_dim)
    pooled_dim = int(transformer.config.pooled_projection_dim)
    context = torch.zeros(batch_size, 0, context_dim, device=device, dtype=dtype)
    pooled = torch.zeros(batch_size, pooled_dim, device=device, dtype=dtype)
    context_ids = torch.zeros(0, 3, device=device, dtype=dtype)
    return context, pooled, context_ids


class LightingTokenEncoder(nn.Module):
    """Encode numeric lighting controls as typed Kontext context tokens."""

    def __init__(
        self,
        schema: LightingSchema,
        context_dim: int,
        hidden_dim: int,
        fourier_features: int,
        fourier_sigma: float,
        fourier_seed: int,
    ):
        super().__init__()
        if context_dim < 1 or hidden_dim < 1 or fourier_features < 1:
            raise ValueError("context_dim, hidden_dim and fourier_features must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(fourier_seed))
        frequencies = torch.randn(
            len(schema.names), fourier_features, generator=generator, dtype=torch.float32
        ) * float(fourier_sigma)
        self.register_buffer("frequencies", frequencies, persistent=True)
        feature_dim = fourier_features * 2 + 2
        self.scalar_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.field_embedding = nn.Parameter(torch.empty(len(schema.names), context_dim))
        self.task_embedding = nn.Embedding(len(TASK_NAMES), context_dim)
        self.type_embedding = nn.Parameter(torch.empty(context_dim))
        self.output_norm = nn.LayerNorm(context_dim)
        self.schema_names = schema.names
        self.context_dim = int(context_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.field_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        for module in self.scalar_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        values: torch.Tensor,
        known: torch.Tensor,
        valid: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        expected = len(self.schema_names)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(f"lighting values must have shape [B, {expected}], got {tuple(values.shape)}")
        if known.shape != values.shape or valid.shape != values.shape:
            raise ValueError("lighting known and valid tensors must match values")
        if task_ids.shape != (values.shape[0],):
            raise ValueError(f"task_ids must have shape [{values.shape[0]}]")

        device = self.field_embedding.device
        values = values.to(device=device, dtype=torch.float32)
        known = known.to(device=device, dtype=torch.float32)
        valid = valid.to(device=device, dtype=torch.float32)
        task_ids = task_ids.to(device=device, dtype=torch.long)
        frequencies = self.frequencies.to(device=device)
        phase = values.unsqueeze(-1) * frequencies.unsqueeze(0) * (2.0 * math.pi)
        features = torch.cat(
            (torch.sin(phase), torch.cos(phase), known.unsqueeze(-1), valid.unsqueeze(-1)), dim=-1
        )
        projection_dtype = self.scalar_projection[0].weight.dtype
        tokens = self.scalar_projection(features.to(dtype=projection_dtype))
        tokens = tokens + self.field_embedding.unsqueeze(0)
        tokens = tokens + self.task_embedding(task_ids).unsqueeze(1)
        tokens = tokens + self.type_embedding.view(1, 1, -1)
        return self.output_norm(tokens)


class LightingConditionedTransformer(nn.Module):
    """Use lighting tokens as FLUX context before joint attention."""

    def __init__(
        self,
        transformer: nn.Module,
        lighting_encoder: LightingTokenEncoder,
        *,
        ddp_trainable_only: bool = False,
    ):
        super().__init__()
        if ddp_trainable_only:
            # The frozen FLUX backbone is loaded independently from the same checkpoint
            # on every rank. Keep it outside this wrapper's registered module tree so
            # DDP does not broadcast billions of frozen parameters during initialization.
            object.__setattr__(self, "transformer", transformer)
            trainable_transformer_parameters = [
                parameter for parameter in transformer.parameters() if parameter.requires_grad
            ]
            if not trainable_transformer_parameters:
                raise RuntimeError("the FLUX transformer has no trainable adapter parameters")
            self.ddp_transformer_parameters = nn.ParameterList(trainable_transformer_parameters)
        else:
            self.transformer = transformer
        self.lighting_encoder = lighting_encoder
        self.ddp_trainable_only = bool(ddp_trainable_only)

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def enable_gradient_checkpointing(self) -> None:
        self.transformer.enable_gradient_checkpointing()

    def forward(
        self,
        *args,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        lighting_values: torch.Tensor | None = None,
        lighting_known: torch.Tensor | None = None,
        lighting_valid: torch.Tensor | None = None,
        lighting_task_ids: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        attention_kwargs = dict(joint_attention_kwargs or {})
        lighting_values = attention_kwargs.pop("tokenlight_values", lighting_values)
        lighting_known = attention_kwargs.pop("tokenlight_known", lighting_known)
        lighting_valid = attention_kwargs.pop("tokenlight_valid", lighting_valid)
        lighting_task_ids = attention_kwargs.pop("tokenlight_task_ids", lighting_task_ids)
        if any(value is None for value in (lighting_values, lighting_known, lighting_valid, lighting_task_ids)):
            raise ValueError("all four lighting conditioning tensors are required")

        lighting_tokens = self.lighting_encoder(
            lighting_values, lighting_known, lighting_valid, lighting_task_ids
        ).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        if lighting_tokens.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("lighting and context batch sizes differ")
        encoder_hidden_states = torch.cat((encoder_hidden_states, lighting_tokens), dim=1)
        lighting_ids = self._lighting_ids(
            txt_ids, lighting_tokens.shape[0], lighting_tokens.shape[1]
        )
        txt_ids = torch.cat((txt_ids, lighting_ids), dim=-2)
        return self.transformer(
            *args,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            joint_attention_kwargs=attention_kwargs or None,
            **kwargs,
        )

    @staticmethod
    def _lighting_ids(txt_ids: torch.Tensor, batch_size: int, token_count: int) -> torch.Tensor:
        if txt_ids.ndim not in (2, 3) or txt_ids.shape[-1] != 3:
            raise ValueError(f"unexpected txt_ids shape: {tuple(txt_ids.shape)}")
        if txt_ids.ndim == 2:
            ids = torch.zeros(token_count, 3, device=txt_ids.device, dtype=txt_ids.dtype)
            ids[:, 0] = 2
            ids[:, 1] = torch.arange(token_count, device=txt_ids.device, dtype=txt_ids.dtype)
            return ids
        ids = torch.zeros(
            batch_size, token_count, 3, device=txt_ids.device, dtype=txt_ids.dtype
        )
        ids[..., 0] = 2
        ids[..., 1] = torch.arange(token_count, device=txt_ids.device, dtype=txt_ids.dtype)
        return ids
