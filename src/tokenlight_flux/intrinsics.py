from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicDecoder(nn.Module):
    """Decode a predicted clean latent into albedo / normal / irradiance maps.

    The input is an **unpacked** VAE latent ``[B, C, H/8, W/8]``; the output
    resolution is set by ``out_resolution`` and defaults to the latent grid
    (a 960 input yields 120).

    This head performs one convolutional decode on the single DiT forward pass;
    it never runs the transformer a second time.
    """

    def __init__(
        self,
        latent_channels: int,
        latent_resolution: int,
        out_resolution: int,
        hidden_channels: int = 256,
    ):
        super().__init__()
        if latent_channels < 1 or hidden_channels < 1:
            raise ValueError("latent_channels and hidden_channels must be positive")
        if min(latent_resolution, out_resolution) < 1:
            raise ValueError("resolutions must be positive")
        if hidden_channels % 8:
            # GroupNorm(8, C) requires the channel count to be a multiple of 8.
            raise ValueError("hidden_channels must be a multiple of 8")
        self.latent_resolution = int(latent_resolution)
        self.out_resolution = int(out_resolution)
        self.hidden_channels = int(hidden_channels)
        self.body = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        blocks: list[nn.Module] = []
        if self.out_resolution > self.latent_resolution:
            if self.out_resolution % self.latent_resolution:
                raise ValueError("out_resolution must be a multiple of the latent resolution")
            ratio = self.out_resolution // self.latent_resolution
            if ratio & (ratio - 1):
                # Each stage doubles the grid, so only powers of two are exact.
                raise ValueError("out_resolution must be a power-of-two multiple of the latent "
                                 "resolution")
            for _ in range(ratio.bit_length() - 1):
                blocks.extend(
                    [
                        nn.Upsample(scale_factor=2.0, mode="nearest"),
                        nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                        nn.SiLU(),
                    ]
                )
        elif self.out_resolution < self.latent_resolution:
            if self.latent_resolution % self.out_resolution:
                raise ValueError("latent resolution must be a multiple of out_resolution")
            blocks.append(nn.AvgPool2d(self.latent_resolution // self.out_resolution))
        self.resample = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(hidden_channels, 3, 3, padding=1)
                for name in ("albedo", "normal", "irradiance")
            }
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 4:
            raise ValueError(f"latent must be 4D [B,C,H,W], got {tuple(latent.shape)}")
        if latent.shape[-2:] != (self.latent_resolution, self.latent_resolution):
            raise ValueError(
                "latent spatial size differs from the decoder: "
                f"{tuple(latent.shape[-2:])} vs {(self.latent_resolution, self.latent_resolution)}"
            )
        feature = self.resample(self.body(latent))
        return {name: head(feature) for name, head in self.heads.items()}


def ain_timestep_weight(sigmas: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Per-sample AIN weight ``w(sigma) = clamp((1-sigma)/tau, 0, 1) ** p``, cut by sigma_max."""
    sigma = sigmas.detach().float().reshape(-1)
    tau = max(float(config.get("sigma_weight_tau", 1.0)), 1e-6)
    power = float(config.get("sigma_weight_power", 1.0))
    weight = ((1.0 - sigma) / tau).clamp(0.0, 1.0) ** power
    sigma_max = float(config.get("sigma_max", 1.0))
    if sigma_max < 1.0:
        weight = weight * (sigma <= sigma_max).float()
    return weight


def intrinsic_losses(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    sample_weight: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """AIN loss over albedo / normal / irradiance.

    The denominator is the weighted count of valid target pixels, not the sum of
    the sample weights, so the scale of the loss does not drift with the batch.
    """
    weights = weights or {"albedo": 1.0, "normal": 1.0, "irradiance": 1.0}
    mask = batch["intrinsic_mask"].float()
    if sample_weight is not None:
        mask = mask * sample_weight.float().view(-1, 1, 1, 1)
    denominator = mask.sum().clamp_min(1.0)

    albedo = (
        (prediction["albedo"].sigmoid() - batch["intrinsic_albedo"]).abs() * mask
    ).sum() / denominator
    cos = (
        F.normalize(prediction["normal"], dim=1)
        * F.normalize(batch["intrinsic_normal"], dim=1)
    ).sum(dim=1, keepdim=True)
    normal = ((1.0 - cos) * mask).sum() / denominator
    irradiance = (
        (prediction["irradiance"].sigmoid() - batch["intrinsic_irradiance"]).abs() * mask
    ).sum() / denominator

    total = (
        float(weights["albedo"]) * albedo
        + float(weights["normal"]) * normal
        + float(weights["irradiance"]) * irradiance
    )
    stats = {
        "albedo": float(albedo.detach()),
        "normal": float(normal.detach()),
        "irradiance": float(irradiance.detach()),
        "total": float(total.detach()),
    }
    return total, stats
