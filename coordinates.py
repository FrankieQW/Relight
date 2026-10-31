"""Camera-space and image-plane coordinates for composition manifests.

This module is the canonical implementation of the convention that the
repository-root ``transfer.py`` script introduced:

``canonical`` frame (written by the composition renderer)
    ``x=right, y=camera-forward, z=up`` with the origin at ``camera.target`` and
    units in meters.  ``canonical.position_axes`` must equal
    :data:`CANONICAL_AXES`.

``camera_position`` (OpenCV camera frame)
    ``[right, down, forward]`` relative to the camera center, i.e.
    ``[x, -z, y + distance]`` where ``distance`` is the camera-to-target
    distance.  This is the vector that training code reads.

``image_plane`` (pixel position plus depth)
    ``u, v`` in pixels with the origin at the top-left corner, plus ``depth``
    equal to ``camera_position[2]``.  ``x``/``y`` store the pinhole normalized
    image coordinates ``X/Z`` and ``Y/Z`` so the 3-D vector can be rebuilt
    without the intrinsics.

``depth`` is the signed distance along the camera forward axis measured from the
camera center.  It is not the Euclidean distance to the camera, and it is not
the distance to the image plane: the image plane is a virtual plane that only
exists together with a focal length, while ``depth`` is defined independently of
it.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

CANONICAL_AXES = "x=right,y=camera-forward,z=up"
CAMERA_AXES = "x=right,y=down,z=forward"
IMAGE_PLANE_PROJECTION = "pinhole"

Vec3 = list[float]


# --------------------------------------------------------------------------- #
# Primitive helpers (kept identical to transfer.py so both agree exactly).
# --------------------------------------------------------------------------- #
def vector(value: Any, name: str) -> Vec3:
    """Validate a three-number JSON value and return it as floats."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain three numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def unit(value: Sequence[float], name: str = "direction") -> Vec3:
    length = math.sqrt(dot(value, value))
    if length < 1e-12:
        raise ValueError(f"{name} cannot define a camera axis")
    return [item / length for item in value]


# --------------------------------------------------------------------------- #
# Camera frame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraFrame:
    """Camera extrinsics derived from a manifest ``camera`` block.

    The basis is the renderer's no-roll convention: ``forward`` points from the
    camera to ``target``, ``right`` is horizontal in the world frame and ``up``
    completes the right-handed triple.
    """

    location: Vec3
    target: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    distance: float

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CameraFrame":
        canonical = row["canonical"]
        if canonical.get("position_axes") != CANONICAL_AXES:
            raise ValueError(
                f"Unsupported canonical.position_axes: {canonical.get('position_axes')!r}"
            )
        camera = row["camera"]
        location = vector(camera["location"], "camera.location")
        target = vector(camera["target"], "camera.target")
        origin = vector(canonical["origin"], "canonical.origin")
        if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(origin, target)):
            raise ValueError("canonical.origin does not match camera.target")
        forward = unit([t - c for t, c in zip(target, location)], "camera forward")
        right = unit(cross(forward, [0.0, 0.0, 1.0]), "camera right")
        up = unit(cross(right, forward), "camera up")
        frame = cls(
            location=location,
            target=target,
            forward=forward,
            right=right,
            up=up,
            distance=math.dist(location, target),
        )
        frame.verify_rotation(camera.get("rotation_euler"))
        return frame

    def verify_rotation(self, rotation_euler: Any) -> None:
        """Confirm the saved Blender XYZ Euler matches the assumed orientation.

        The renderer always aims the camera at ``target`` without roll; a
        disagreement means the manifest was produced by different code and the
        simple linear conversion would silently produce wrong coordinates.
        """
        if rotation_euler is None:
            return
        rx, ry, rz = vector(rotation_euler, "camera.rotation_euler")
        sx, cx = math.sin(rx), math.cos(rx)
        sy, cy = math.sin(ry), math.cos(ry)
        sz, cz = math.sin(rz), math.cos(rz)
        actual_right = [cz * cy, sz * cy, -sy]
        actual_up = [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx]
        actual_forward = [
            -(cz * sy * cx + sz * sx),
            -(sz * sy * cx - cz * sx),
            -cy * cx,
        ]
        for name, expected, actual in (
            ("right", self.right, actual_right),
            ("up", self.up, actual_up),
            ("forward", self.forward, actual_forward),
        ):
            if max(abs(a - b) for a, b in zip(expected, actual)) > 1e-5:
                raise ValueError(
                    "Camera rotation does not match the renderer's target-facing, "
                    f"no-roll convention ({name} axis)"
                )

    # -- canonical <-> camera ------------------------------------------------
    def canonical_to_camera(self, point: Sequence[float]) -> Vec3:
        """``[x, y, z]`` canonical -> ``[right, down, forward]`` camera."""
        if len(point) != 3:
            raise ValueError("point must contain three numbers")
        x, y, z = (float(value) for value in point)
        return [x, -z, y + self.distance]

    def camera_to_canonical(self, point: Sequence[float]) -> Vec3:
        """``[right, down, forward]`` camera -> ``[x, y, z]`` canonical."""
        if len(point) != 3:
            raise ValueError("point must contain three numbers")
        right_offset, down_offset, depth = (float(value) for value in point)
        return [right_offset, depth - self.distance, -down_offset]

    # -- world <-> camera ----------------------------------------------------
    def world_to_camera(self, point: Sequence[float]) -> Vec3:
        """Blender world meters -> ``[right, down, forward]`` camera."""
        if len(point) != 3:
            raise ValueError("point must contain three numbers")
        delta = [float(p) - c for p, c in zip(point, self.location)]
        return [dot(delta, self.right), -dot(delta, self.up), dot(delta, self.forward)]

    def camera_to_world(self, point: Sequence[float]) -> Vec3:
        """``[right, down, forward]`` camera -> Blender world meters."""
        if len(point) != 3:
            raise ValueError("point must contain three numbers")
        right_offset, down_offset, depth = (float(value) for value in point)
        return [
            c + right_offset * r - down_offset * u + depth * f
            for c, r, u, f in zip(self.location, self.right, self.up, self.forward)
        ]


# --------------------------------------------------------------------------- #
# Pinhole intrinsics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for the rendered image.

    ``u = fx * X / Z + cx`` and ``v = fy * Y / Z + cy`` where ``X`` is right,
    ``Y`` is down and ``Z`` is the forward depth, so the formulas apply directly
    to ``camera_position``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera(
        cls,
        camera: dict[str, Any],
        *,
        resolution: tuple[int, int] | None = None,
    ) -> "Intrinsics | None":
        """Resolve intrinsics from a manifest camera block.

        ``resolution`` overrides the pixel size when the manifest predates the
        intrinsics fields.  Returns ``None`` when the block does not describe
        enough information to project, which lets callers decide whether pixel
        coordinates are mandatory.  The pinhole parameters are the single source
        of truth: a caller that only knows the resolution but not the field of
        view still gets ``None`` instead of a silently wrong guess.
        """
        block = camera.get("intrinsics")
        if isinstance(block, dict):
            width = int(block.get("width") or 0)
            height = int(block.get("height") or 0)
            if width > 0 and height > 0:
                return cls(
                    width=width,
                    height=height,
                    fx=float(block["fx"]),
                    fy=float(block["fy"]),
                    cx=float(block["cx"]),
                    cy=float(block["cy"]),
                )
        width, height = _resolve_resolution(camera, resolution)
        if width <= 0 or height <= 0:
            return None
        angle_x = _angle(camera, "angle_x", width, height)
        angle_y = _angle(camera, "angle_y", width, height)
        if angle_x is None or angle_y is None:
            return None
        shift_x = float(camera.get("shift_x", 0.0) or 0.0)
        shift_y = float(camera.get("shift_y", 0.0) or 0.0)
        fx = (width * 0.5) / math.tan(angle_x * 0.5)
        fy = (height * 0.5) / math.tan(angle_y * 0.5)
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=width * 0.5 - width * shift_x,
            cy=height * 0.5 + height * shift_y,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """Camera ``[right, down, forward]`` -> pixel ``(u, v)``."""
        if len(point) != 3:
            raise ValueError("point must contain three numbers")
        x, y, depth = (float(value) for value in point)
        if depth == 0.0:
            raise ValueError("cannot project a point on the camera center plane")
        return (self.fx * x / depth + self.cx, self.fy * y / depth + self.cy)

    def unproject(self, x_normalized: float, y_normalized: float, depth: float) -> Vec3:
        """Normalized image coordinates plus depth -> camera point."""
        return [x_normalized * depth, y_normalized * depth, depth]

    def pixel_to_normalized(self, u: float, v: float) -> tuple[float, float]:
        return ((u - self.cx) / self.fx, (v - self.cy) / self.fy)


def _resolve_resolution(
    camera: dict[str, Any], override: tuple[int, int] | None
) -> tuple[int, int]:
    if override is not None:
        return (int(override[0]), int(override[1]))
    block = camera.get("resolution")
    if isinstance(block, dict):
        return (int(block.get("width") or 0), int(block.get("height") or 0))
    if isinstance(block, (list, tuple)) and len(block) == 2:
        return (int(block[0]), int(block[1]))
    return (0, 0)


def _angle(camera: dict[str, Any], key: str, width: int, height: int) -> float | None:
    value = camera.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    lens = camera.get("focal_length") or camera.get("lens")
    sensor = camera.get("sensor_width")
    if not isinstance(lens, (int, float)) or not isinstance(sensor, (int, float)):
        return None
    if float(lens) <= 0 or float(sensor) <= 0:
        return None
    fit = str(camera.get("sensor_fit", "AUTO")).upper()
    horizontal = 2.0 * math.atan(float(sensor) / (2.0 * float(lens)))
    if fit in {"HORIZONTAL", "AUTO"} or width >= height:
        return horizontal if key == "angle_x" else 2.0 * math.atan(
            math.tan(horizontal * 0.5) * height / width
        )
    vertical = 2.0 * math.atan(float(sensor) / (2.0 * float(lens)))
    return vertical if key == "angle_y" else 2.0 * math.atan(
        math.tan(vertical * 0.5) * width / height
    )


# --------------------------------------------------------------------------- #
# Image-plane representation
# --------------------------------------------------------------------------- #
def image_plane_point(
    camera_point: Sequence[float],
    intrinsics: Intrinsics | None,
) -> dict[str, Any]:
    """Describe a camera-space point on the image plane plus its depth.

    Always returns the normalized pinhole coordinates ``x = X/Z`` and ``y = Y/Z``
    and the ``depth = Z`` term, which are defined without intrinsics.  Pixel
    ``u``/``v`` and the ``[0, 1]`` NDC copy are added when intrinsics exist.
    """
    if len(camera_point) != 3:
        raise ValueError("camera point must contain three numbers")
    right_offset, down_offset, depth = (float(value) for value in camera_point)
    if depth == 0.0:
        raise ValueError("depth is zero: the light lies on the camera center plane")
    x_normalized = right_offset / depth
    y_normalized = down_offset / depth
    result: dict[str, Any] = {
        "x": x_normalized,
        "y": y_normalized,
        "depth": depth,
        "in_front": depth > 0.0,
        "projection": IMAGE_PLANE_PROJECTION,
    }
    if intrinsics is not None:
        u, v = intrinsics.project(camera_point)
        result.update(
            {
                "u": u,
                "v": v,
                "x_ndc": u / intrinsics.width,
                "y_ndc": v / intrinsics.height,
                "width": intrinsics.width,
                "height": intrinsics.height,
            }
        )
    return result


def image_plane_to_camera(
    x_normalized: float,
    y_normalized: float,
    depth: float,
) -> Vec3:
    """Rebuild ``[right, down, forward]`` from normalized image coordinates."""
    return [float(x_normalized) * float(depth), float(y_normalized) * float(depth), float(depth)]


# --------------------------------------------------------------------------- #
# Manifest conversion
# --------------------------------------------------------------------------- #
CONVENTION = {
    "coordinate_space": "opencv_camera",
    "origin": "camera.location",
    "axes": CAMERA_AXES,
    "depth": "signed_distance_along_camera_forward_axis",
    "image_plane": "pinhole_u_v_pixels_top_left_origin",
    "position_formula": "camera_position = [canonical_x, -canonical_z, canonical_y + distance]",
}


def convert_row(
    row: dict[str, Any],
    *,
    resolution: tuple[int, int] | None = None,
    require_intrinsics: bool = True,
) -> tuple[dict[str, Any], int, bool]:
    """Add camera coordinates to every light of one manifest row.

    Returns ``(row, light_count, intrinsics_available)``.  ``camera_position`` is
    always written; ``image_plane`` is written whenever intrinsics can be
    resolved.  When ``require_intrinsics`` is true a manifest that cannot supply
    them raises instead of silently emitting depth-only rows.
    """
    if not isinstance(row, dict):
        raise ValueError("Manifest row must be a JSON object")
    frame = CameraFrame.from_row(row)
    camera = row["camera"]
    intrinsics = Intrinsics.from_camera(camera, resolution=resolution)
    if intrinsics is None and require_intrinsics:
        raise ValueError(
            "camera block has no projection parameters; re-render with intrinsics or pass "
            "--resolution WIDTHxHEIGHT"
        )
    canonical = row["canonical"]
    light_count = 0
    for field in ("point_lights", "in_scene_lights"):
        lights = row.get(field, [])
        if not isinstance(lights, list):
            raise ValueError(f"{field} must be a list")
        for index, light in enumerate(lights):
            if not isinstance(light, dict):
                raise ValueError(f"{field}[{index}] must be an object")
            converted = frame.canonical_to_camera(vector(light["position"], f"{field}[{index}].position"))
            if "renderer_position" in light:
                world = vector(light["renderer_position"], f"{field}[{index}].renderer_position")
                reference = frame.world_to_camera(world)
                if not all(
                    math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(converted, reference)
                ):
                    raise ValueError(f"{field}[{index}] position disagrees with renderer_position")
            light["camera_position"] = converted
            light["image_plane"] = image_plane_point(converted, intrinsics)
            light_count += 1
    convention = dict(CONVENTION)
    convention["units"] = canonical.get("units", "same_as_renderer_position")
    if intrinsics is not None:
        convention["intrinsics"] = intrinsics.to_dict()
    row["camera_position_convention"] = convention
    return row, light_count, intrinsics is not None


def convert_rows(
    rows: Iterable[dict[str, Any]],
    *,
    resolution: tuple[int, int] | None = None,
    require_intrinsics: bool = True,
) -> list[dict[str, Any]]:
    """Convert rows in memory; used by tests and notebooks."""
    return [
        convert_row(row, resolution=resolution, require_intrinsics=require_intrinsics)[0]
        for row in rows
    ]


def parse_resolution(value: str | None) -> tuple[int, int] | None:
    """Parse ``960``, ``960x960`` or ``960,540`` into ``(width, height)``."""
    if not value:
        return None
    text = str(value).strip().lower().replace("x", " ").replace(",", " ").split()
    numbers = [int(item) for item in text if item]
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    if len(numbers) == 2 and all(number > 0 for number in numbers):
        return (numbers[0], numbers[1])
    raise ValueError(f"invalid resolution: {value!r} (use 960 or 960x540)")


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    resolution: tuple[int, int] | None = None,
    require_intrinsics: bool = True,
    overwrite: bool = False,
) -> list[tuple[str, int, int, bool]]:
    """Convert every top-level ``*.jsonl`` manifest and write new files.

    Mirrors ``transfer.py``: the output directory must differ from the input and
    existing files are not overwritten unless ``overwrite`` is set.  Files are
    staged in a temporary directory on the same filesystem before publishing, so
    a failure never leaves a half-written manifest behind.
    """
    source_dir = Path(input_dir).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {source_dir}")
    if source_dir == target_dir:
        raise ValueError("Input and output directories must be different")
    sources = sorted(source_dir.glob("*.jsonl"))
    if not sources:
        raise ValueError(f"No top-level *.jsonl files found in {source_dir}")
    if target_dir.exists() and not target_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {target_dir}")
    for source in sources:
        destination = target_dir / source.name
        if destination.exists() and not overwrite:
            raise ValueError(f"Refusing to overwrite existing output: {destination}")
    target_dir.mkdir(parents=True, exist_ok=True)
    stats: list[tuple[str, int, int, bool]] = []
    with tempfile.TemporaryDirectory(prefix=".coordinates-", dir=target_dir) as temporary:
        staging = Path(temporary)
        for source in sources:
            rows = lights = 0
            intrinsics_seen = False
            with source.open("r", encoding="utf-8-sig") as reader, (
                staging / source.name
            ).open("w", encoding="utf-8", newline="\n") as writer:
                for line_number, line in enumerate(reader, 1):
                    if not line.strip():
                        continue
                    try:
                        row, count, has_intrinsics = convert_row(
                            json.loads(line, parse_constant=reject_constant),
                            resolution=resolution,
                            require_intrinsics=require_intrinsics,
                        )
                    except (KeyError, TypeError, ValueError, OverflowError) as error:
                        raise ValueError(f"{source.name}:{line_number}: {error}") from error
                    writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows += 1
                    lights += count
                    intrinsics_seen = intrinsics_seen or has_intrinsics
            stats.append((source.name, rows, lights, intrinsics_seen))
        for source in sources:
            destination = target_dir / source.name
            with destination.open("w", encoding="utf-8", newline="\n") as writer:
                with (staging / source.name).open("r", encoding="utf-8") as reader:
                    for line in reader:
                        writer.write(line)
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add OpenCV camera coordinates plus image-plane pixels and depth to "
            "composition manifests."
        )
    )
    parser.add_argument("input_dir", help="Original manifests directory")
    parser.add_argument("output_dir", help="Output directory for converted manifests")
    parser.add_argument(
        "--resolution",
        default=None,
        help="Pixel size for manifests without intrinsics, e.g. 960 or 960x540",
    )
    parser.add_argument(
        "--allow-missing-intrinsics",
        action="store_true",
        help="Write depth-only image_plane entries instead of failing",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args(argv)
    try:
        stats = convert_directory(
            args.input_dir,
            args.output_dir,
            resolution=parse_resolution(args.resolution),
            require_intrinsics=not args.allow_missing_intrinsics,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    for name, rows, lights, intrinsics in stats:
        suffix = "with image-plane pixels" if intrinsics else "depth only"
        print(f"{name}: {rows} rows, {lights} lights converted ({suffix})")
    print("Done. Coordinates: OpenCV camera (X right, Y down, Z forward) + image plane + depth.")
    print("Original position is preserved. Training must read camera_position or image_plane.")


if __name__ == "__main__":
    main()
