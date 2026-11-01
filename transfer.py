"""Convert rendered manifest light positions to OpenCV camera coordinates.

Usage:
    python transfer.py /path/to/manifests /path/to/manifests_camera

Reads every top-level *.jsonl file in the input directory. Original fields are
preserved; each point/fixture light receives camera_position = [right, down,
forward], relative to the camera center. Training must read camera_position
instead of position to use these coordinates. No third-party packages required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


EXPECTED_AXES = "x=right,y=camera-forward,z=up"


def vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain three numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def unit(value):
    length = math.sqrt(dot(value, value))
    if length < 1e-12:
        raise ValueError("Camera direction cannot define the expected canonical frame")
    return [item / length for item in value]


def convert_row(row):
    if not isinstance(row, dict):
        raise ValueError("Manifest row must be a JSON object")
    canonical = row["canonical"]
    if canonical.get("position_axes") != EXPECTED_AXES:
        raise ValueError(f"Unsupported canonical.position_axes: {canonical.get('position_axes')!r}")
    camera = row["camera"]
    location = vector(camera["location"], "camera.location")
    target = vector(camera["target"], "camera.target")
    origin = vector(canonical["origin"], "canonical.origin")
    if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(origin, target)):
        raise ValueError("canonical.origin does not match camera.target")
    forward = unit([t - c for t, c in zip(target, location)])
    right = unit(cross(forward, [0.0, 0.0, 1.0]))
    up = unit(cross(right, forward))
    distance = math.dist(location, target)

    # Blender's default XYZ Euler rotation is Rz @ Ry @ Rx. Verify that the
    # saved camera actually faces target with the no-roll orientation assumed
    # by the renderer's canonical frame before applying the simple formula.
    rx, ry, rz = vector(camera["rotation_euler"], "camera.rotation_euler")
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    actual_right = [cz * cy, sz * cy, -sy]
    actual_up = [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx]
    actual_forward = [-(cz * sy * cx + sz * sx), -(sz * sy * cx - cz * sx), -cy * cx]
    for expected, actual in ((right, actual_right), (up, actual_up), (forward, actual_forward)):
        if max(abs(a - b) for a, b in zip(expected, actual)) > 1e-5:
            raise ValueError("Camera rotation does not match the renderer's target-facing, no-roll convention")

    light_count = 0
    for field in ("point_lights", "in_scene_lights"):
        lights = row.get(field, [])
        if not isinstance(lights, list):
            raise ValueError(f"{field} must be a list")
        for index, light in enumerate(lights):
            if not isinstance(light, dict):
                raise ValueError(f"{field}[{index}] must be an object")
            x, y, z = vector(light["position"], f"{field}[{index}].position")
            converted = [x, -z, y + distance]
            if "renderer_position" in light:
                world = vector(light["renderer_position"], f"{field}[{index}].renderer_position")
                delta = [p - c for p, c in zip(world, location)]
                reference = [dot(delta, right), -dot(delta, up), dot(delta, forward)]
                if not all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                           for a, b in zip(converted, reference)):
                    raise ValueError(f"{field}[{index}] position disagrees with renderer_position")
            light["camera_position"] = converted
            light_count += 1

    row["camera_position_convention"] = {
        "coordinate_space": "opencv_camera",
        "origin": "camera.location",
        "axes": "x=right,y=down,z=forward",
        "units": canonical.get("units", "same_as_renderer_position"),
        "depth": "signed_distance_along_camera_forward_axis",
    }
    return row, light_count


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def convert_directory(input_dir, output_dir):
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    sources = sorted(input_dir.glob("*.jsonl"))
    if not sources:
        raise ValueError(f"No top-level *.jsonl files found in {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    for source in sources:
        if (output_dir / source.name).exists():
            raise ValueError(f"Refusing to overwrite existing output: {output_dir / source.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    # Validate and stage every split before publishing any final manifest.
    # Temporary files live on the same filesystem as the output directory.
    with tempfile.TemporaryDirectory(prefix=".transfer-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for source in sources:
            rows = lights = 0
            with source.open("r", encoding="utf-8-sig") as reader, (staging / source.name).open(
                "w", encoding="utf-8", newline="\n"
            ) as writer:
                for line_number, line in enumerate(reader, 1):
                    if not line.strip():
                        continue
                    try:
                        row, count = convert_row(json.loads(line, parse_constant=reject_constant))
                        writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    except (KeyError, TypeError, ValueError, OverflowError) as error:
                        raise ValueError(f"{source.name}:{line_number}: {error}") from error
                    rows += 1
                    lights += count
            stats.append((source.name, rows, lights))
        for source in sources:
            # Exclusive creation protects existing files even if another process
            # created one after the initial checks.
            destination = output_dir / source.name
            with destination.open("x", encoding="utf-8", newline="\n") as writer:
                try:
                    with (staging / source.name).open("r", encoding="utf-8") as reader:
                        for line in reader:
                            writer.write(line)
                except BaseException:
                    writer.close()
                    destination.unlink(missing_ok=True)
                    raise
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Original manifests directory")
    parser.add_argument("output_dir", help="Output directory for converted manifests")
    args = parser.parse_args()
    try:
        stats = convert_directory(args.input_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    for name, rows, lights in stats:
        print(f"{name}: {rows} rows, {lights} lights converted")
    print("Done. Coordinates: OpenCV camera (X right, Y down, Z forward).")
    print("Original position is preserved. Training must read camera_position to use the conversion.")


if __name__ == "__main__":
    main()
