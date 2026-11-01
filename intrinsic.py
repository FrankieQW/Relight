"""Intrinsic channels derived from Cycles passes: albedo, irradiance, normals.

Everything here is plain NumPy so the maths can be tested without Blender,
without a GPU and without OpenCV.  The Blender-side renderer only produces the
raw Cycles passes; these functions turn them into the dataset's intrinsic maps
and decide whether the material pipeline actually delivered textures.

Definitions (also documented in ``datacompose/GUID.md``)::

    albedo(p)        = Cycles Diffuse Color(p)                     # linear, unlit
    irradiance_L(p)  = (DiffuseDirect_L + DiffuseIndirect_L)(p) / albedo(p)   # E/pi
    normal_camera(p) = R_cam^T . N_world(p)                        # unit, X right/Y down/Z forward
    valid(p)         = albedo(p) >= eps  and  ||N_world(p)|| > 0

``irradiance`` is the cosine-weighted irradiance divided by pi, i.e. what a white
Lambertian surface would reflect under the same lighting.  Pixels whose albedo is
below ``eps`` (metals, glass) carry no information and are marked invalid instead
of being guessed.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "albedo_quality",
    "derive_irradiance",
    "evaluate_material_audit",
    "local_variance",
    "valid_mask",
    "world_normals_to_camera",
]

DEFAULT_EPS = 1e-3


def _as_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"{name} must be an HxWx3 image, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array[..., :3]


def derive_irradiance(
    diffuse_color: Any,
    diffuse_direct: Any,
    diffuse_indirect: Any,
    *,
    eps: float = DEFAULT_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(irradiance, valid)`` with ``E/pi = (direct + indirect) / albedo``.

    ``valid`` is per pixel and requires *every* albedo channel to reach ``eps``;
    those pixels get irradiance 0 so downstream code never has to handle NaN.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    albedo = _as_image(diffuse_color, "diffuse_color")
    direct = _as_image(diffuse_direct, "diffuse_direct")
    indirect = _as_image(diffuse_indirect, "diffuse_indirect")
    if albedo.shape != direct.shape or albedo.shape != indirect.shape:
        raise ValueError(
            f"pass shapes disagree: {albedo.shape}, {direct.shape}, {indirect.shape}"
        )
    valid = np.all(albedo >= eps, axis=2)
    safe = np.where(valid[..., None], albedo, 1.0)
    irradiance = np.where(valid[..., None], (direct + indirect) / safe, 0.0)
    irradiance = np.nan_to_num(irradiance, nan=0.0, posinf=0.0, neginf=0.0)
    return irradiance.astype(np.float32), valid


def world_normals_to_camera(
    world_normals: Any,
    frame: Any,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Convert world-space normals to the OpenCV camera frame.

    ``frame`` must expose ``right``, ``up`` and ``forward`` (see
    :class:`datacompose.coordinates.CameraFrame`).  Background pixels (zero
    normals) stay zero so a caller can tell geometry from empty space.
    """
    normals = _as_image(world_normals, "world_normals")
    basis = np.stack(
        [
            np.asarray(frame.right, dtype=np.float32),
            np.asarray(frame.up, dtype=np.float32),
            np.asarray(frame.forward, dtype=np.float32),
        ]
    )
    if basis.shape != (3, 3):
        raise ValueError("camera frame basis must be three 3-vectors")
    # OpenCV: X right, Y down, Z forward.  Blender's up is +Y in camera space,
    # so the vertical component flips sign.
    signs = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    camera_normals = np.tensordot(normals, basis.T, axes=([2], [0])) * signs
    if normalize:
        length = np.linalg.norm(camera_normals, axis=2, keepdims=True)
        np.divide(camera_normals, length, out=camera_normals, where=length > 1e-6)
    return camera_normals.astype(np.float32)


def valid_mask(
    albedo: Any,
    world_normals: Any,
    *,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Pixels with a usable albedo and actual geometry behind them."""
    albedo_image = _as_image(albedo, "albedo")
    normals = _as_image(world_normals, "world_normals")
    if albedo_image.shape != normals.shape:
        raise ValueError(f"albedo/normal shapes disagree: {albedo_image.shape} vs {normals.shape}")
    geometry = np.linalg.norm(normals, axis=2) > 0.5
    return np.all(albedo_image >= eps, axis=2) & geometry


def local_variance(image: Any, window: int = 5) -> np.ndarray:
    """Per-pixel mean of the local variance across channels (box filter)."""
    array = _as_image(image, "image")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd number >= 3")
    mean = _box_mean(array, window)
    mean_square = _box_mean(array * array, window)
    return np.clip(mean_square - mean * mean, 0.0, None).max(axis=2)


def _box_mean(image: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    height, width = image.shape[:2]
    zero_row = np.zeros((1, cumulative.shape[1], cumulative.shape[2]), dtype=cumulative.dtype)
    cumulative = np.concatenate([zero_row, cumulative], axis=0)
    zero_column = np.zeros((cumulative.shape[0], 1, cumulative.shape[2]), dtype=cumulative.dtype)
    cumulative = np.concatenate([zero_column, cumulative], axis=1)
    total = (
        cumulative[size : size + height, size : size + width]
        - cumulative[:height, size : size + width]
        - cumulative[size : size + height, :width]
        + cumulative[:height, :width]
    )
    return total / float(size * size)


def albedo_quality(
    albedo: Any,
    *,
    local_window: int = 5,
    flat_threshold: float = 1e-6,
    magenta_tolerance: float = 0.01,
) -> dict[str, float]:
    """Summarise whether an albedo image actually carries textures.

    ``magenta_ratio`` catches Blender's missing-texture placeholder, while
    ``textured_pixel_ratio`` separates a properly textured render from one where
    every material fell back to a flat colour.
    """
    image = _as_image(albedo, "albedo")
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    magenta = (red > 1.0 - magenta_tolerance) & (green < magenta_tolerance) & (
        blue > 1.0 - magenta_tolerance
    )
    variance = local_variance(image, local_window)
    textured = variance > flat_threshold
    flat = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8).reshape(-1, 3)
    # 512^2 and 960^2 both produce a lot of rows; sampling keeps this cheap and
    # the count is only used as an order-of-magnitude signal.
    if flat.shape[0] > 200_000:
        flat = flat[:: max(1, flat.shape[0] // 200_000)]
    unique_colors = int(np.unique(flat, axis=0).shape[0])
    return {
        "magenta_ratio": float(magenta.mean()),
        "textured_pixel_ratio": float(textured.mean()),
        "unique_colors": unique_colors,
        "mean": float(image.mean()),
        "std": float(image.std()),
        "max_local_variance": float(variance.max()),
    }


def evaluate_material_audit(
    report: Mapping[str, Any],
    *,
    min_base_color_linked_ratio: float = 0.9,
    require_textures: bool = True,
    max_missing_images: int = 0,
) -> list[str]:
    """Return human-readable problems for a material audit report.

    Empty list means the material pipeline looks usable.  When
    ``require_textures`` is false the audit is advisory: nothing is considered a
    violation, so the caller can downgrade the findings to warnings.  Callers
    decide whether a returned problem is fatal.
    """
    if not require_textures:
        return []
    problems: list[str] = []
    missing = int(report.get("missing_image_files") or 0)
    if missing > int(max_missing_images):
        examples = report.get("missing_examples") or []
        suffix = f"; e.g. {examples[:5]}" if examples else ""
        problems.append(f"{missing} material image file(s) are missing on disk{suffix}")

    textured = int(report.get("textured_materials") or report.get("textured") or 0)
    ratio = report.get("base_color_linked_ratio")
    if textured > 0:
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            problems.append("material audit has no usable base_color_linked_ratio")
        elif float(ratio) < float(min_base_color_linked_ratio):
            problems.append(
                f"only {float(ratio):.1%} of textured materials link Base Color to an image "
                f"(threshold {float(min_base_color_linked_ratio):.0%})"
            )
    return problems
