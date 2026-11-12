"""Scene-only TokenLight renderer (Blender side).

One job renders one existing scene from one camera pose with a controlled
lighting rig, and writes the same linear components as before:

``ambient.exr`` / ``dark.exr`` / ``diffuse/*.exr`` / ``point_lights/*.exr`` /
``in_scene_lights/*.exr`` (+ optional albedo / normal / irradiance passes).

There is no object insertion: the camera pose is chosen upstream by
``scene_cameras.py`` (AABB broad phase) and is **re-validated here against real
triangles** before anything is rendered, so a pose that would clip through
geometry fails the job instead of producing a broken sample.

Run by the scheduler (``render_scenes.py``), never imported by the package:

    blender --background --factory-startup --python render_scene.py -- \
        --runtime-config <snapshot.json>
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
from typing import Any

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    runtime = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    helpers = _load_helpers()
    recovery = _load_recovery()
    jobs = _read_jsonl(Path(runtime["jobs_path"]))
    worker_index = int(runtime["worker_index"])
    worker_count = int(runtime["worker_count"])
    selected = [job for index, job in enumerate(jobs) if index % worker_count == worker_index]
    output_root = Path(runtime["output_root"])
    worker_root = output_root / "render_workers" / f"worker_{worker_index:03d}"
    errors: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for job in selected:
        try:
            metadata = render_job(job, runtime, helpers, recovery)
            completed.append({"job_id": job["job_id"], "metadata": metadata})
            print(f"SCENE_DONE {job['job_id']}", flush=True)
        except Exception as error:  # noqa: BLE001 - every failed render stays recoverable by job id.
            errors.append(
                {
                    "job_id": job.get("job_id"),
                    "error": type(error).__name__,
                    "detail": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"SCENE_FAILED {job.get('job_id')}: {error}", flush=True)
    _write_jsonl(worker_root / "completed.jsonl", completed)
    _write_jsonl(worker_root / "errors.jsonl", errors)
    if errors and not bool(runtime.get("allow_partial", False)):
        raise RuntimeError(f"worker {worker_index} failed {len(errors)} render job(s)")


def render_job(job: dict[str, Any], runtime: dict[str, Any], helpers, recovery) -> str:
    output_root = Path(runtime["output_root"])
    final_directory = output_root / "components" / job["job_id"]
    metadata_path = final_directory / "metadata.json"
    if metadata_path.is_file() and not bool(runtime.get("overwrite", False)):
        _validate_reusable_metadata(metadata_path, job)
        return metadata_path.relative_to(output_root).as_posix()
    if final_directory.exists():
        if not bool(runtime.get("overwrite", False)):
            raise FileExistsError(f"incomplete output exists and overwrite=false: {final_directory}")
        shutil.rmtree(final_directory)

    project_root = Path(runtime["project_root"])
    blend_path = (project_root / job["base_scene_blend"]).resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(blend_path)
    if _sha256_file(blend_path) != job["base_scene_digest"]:
        raise ValueError("base scene blend digest changed after camera planning")

    # Preflight passed: only now is a partial directory allowed to exist.
    partial = recovery.prepare_render_partial(output_root, job["job_id"])
    try:
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        render_config = _render_config(runtime.get("render", {}))
        helpers.configure_render(render_config)
        passes_config = render_config.get("passes") or {}
        passes_enabled = bool(passes_config.get("enabled", False))
        intrinsic = coordinates = None
        if passes_enabled:
            helpers.configure_passes(render_config)
            intrinsic = _load_intrinsic()
            coordinates = _load_coordinates()

        gate = _validate_camera_pose(job, runtime)
        camera = _apply_camera(job, render_config, gate)
        bpy.context.scene.camera = camera
        anchor = Vector(tuple(float(value) for value in job["camera"]["target"]))

        _disable_native_lighting()
        background, ambient_hdri = _configure_ambient_world(render_config, job)

        passes_ctx = None
        quality = None
        material_audit = None
        if passes_enabled:
            passes_dir = partial / "passes"
            passes_dir.mkdir(parents=True, exist_ok=True)
            material_audit = helpers.audit_materials()
            (passes_dir / "material_audit.json").write_text(
                json.dumps(material_audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            problems = intrinsic.evaluate_material_audit(
                material_audit,
                min_base_color_linked_ratio=float(
                    passes_config.get("min_base_color_linked_ratio", 0.9)
                ),
                require_textures=bool(passes_config.get("require_textures", False)),
            )
            if problems:
                raise ValueError(
                    "material audit gate failed (require_textures=true): " + "; ".join(problems)
                )
            eps = float(passes_config.get("eps", 0.001))
            reference = helpers.render_pass_reference(passes_dir, passes_config)
            albedo_image = helpers.read_linear_image(reference["albedo"])
            normal_world = helpers.read_linear_image(reference["normal_world"])
            frame = coordinates.CameraFrame.from_row(
                {
                    "canonical": {
                        "origin": [float(value) for value in anchor],
                        "position_axes": "x=right,y=camera-forward,z=up",
                    },
                    "camera": {
                        "location": [float(value) for value in camera.location],
                        "target": [float(value) for value in anchor],
                        "rotation_euler": [float(value) for value in camera.rotation_euler],
                    },
                }
            )
            normal_camera = intrinsic.world_normals_to_camera(normal_world, frame)
            valid = intrinsic.valid_mask(albedo_image, normal_world, eps=eps)
            quality = intrinsic.albedo_quality(albedo_image)
            helpers.save_linear_image(passes_dir / "normal_camera.exr", normal_camera)
            helpers.save_mask_image(passes_dir / "valid_mask.png", valid.astype(np.float32))
            if not bool(passes_config.get("save_normal_world", False)):
                (passes_dir / "normal_world.exr").unlink(missing_ok=True)
            passes_ctx = {
                "enabled": True,
                "dir": passes_dir,
                "config": passes_config,
                "eps": eps,
                "irradiance": {},
            }

        ambient_path = partial / "ambient.exr"
        background.inputs["Strength"].default_value = float(render_config["ambient_world_strength"])
        if passes_ctx is not None and bool(passes_config.get("irradiance", True)):
            outcome = helpers.render_exr(
                ambient_path,
                passes_ctx["dir"],
                ("irradiance",),
                name="ambient",
                eps=passes_ctx["eps"],
            )
            if "irradiance" in outcome:
                passes_ctx["irradiance"]["ambient"] = _future_relative(
                    outcome["irradiance"], partial, final_directory, output_root
                )
        else:
            helpers.render_exr(ambient_path)
        background.inputs["Strength"].default_value = 0.0
        dark_path = partial / "dark.exr"
        helpers.render_exr(dark_path)

        rng = random.Random(int(job["seed"]))
        anchor_depth = max(float(job["canonical"]["asset_size"]), 0.1)
        fixture = _select_native_fixture(job, runtime, camera, anchor, partial, helpers, rng)
        fixture_component = _render_fixture_component(
            fixture, partial, output_root, final_directory, render_config, helpers, anchor, passes_ctx
        )
        point_components = _render_point_components(
            partial,
            output_root,
            final_directory,
            camera,
            anchor,
            anchor_depth,
            render_config,
            helpers,
            rng,
            passes_ctx,
        )
        diffuse_components = _render_diffuse_components(
            partial, output_root, final_directory, camera, anchor, anchor_depth, render_config, helpers
        )

        passes_metadata = None
        if passes_ctx is not None:
            passes_metadata = {
                "definition": {
                    "albedo": "cycles_diffuse_color",
                    "irradiance": "diffuse_radiance_over_albedo",
                    "normal": "camera_space_unit_opencv",
                    "eps": float(passes_ctx["eps"]),
                },
                "albedo": _future_relative(
                    passes_dir / "albedo.exr", partial, final_directory, output_root
                ),
                "normal_camera": _future_relative(
                    passes_dir / "normal_camera.exr", partial, final_directory, output_root
                ),
                "valid_mask": _future_relative(
                    passes_dir / "valid_mask.png", partial, final_directory, output_root
                ),
                "material_audit": _future_relative(
                    passes_dir / "material_audit.json", partial, final_directory, output_root
                ),
                "quality": {
                    "base_color_linked_ratio": (material_audit or {}).get("base_color_linked_ratio"),
                    "magenta_ratio": float(quality["magenta_ratio"]),
                    "textured_pixel_ratio": float(quality["textured_pixel_ratio"]),
                },
                "irradiance": dict(passes_ctx["irradiance"]),
            }
            if bool(passes_config.get("save_normal_world", False)):
                passes_metadata["normal_world"] = _future_relative(
                    passes_dir / "normal_world.exr", partial, final_directory, output_root
                )

        metadata = {
            "schema_version": job["schema_version"],
            "id": job["job_id"],
            "scene_id": job["scene_id"],
            "base_scene_blend": job["base_scene_blend"],
            "base_scene_digest": job["base_scene_digest"],
            "base_scene_fingerprint": _scene_fingerprint(),
            "ambient": _future_relative(ambient_path, partial, final_directory, output_root),
            "hdri": str(ambient_hdri) if ambient_hdri is not None else None,
            "ambient_world": {
                "source": "hdri" if ambient_hdri is not None else "constant_color",
                "strength": float(render_config["ambient_world_strength"]),
                "color": None
                if ambient_hdri is not None
                else [float(value) for value in render_config["ambient_world_color"]],
            },
            "dark": _future_relative(dark_path, partial, final_directory, output_root),
            "point_lights": point_components,
            "diffuse": diffuse_components,
            "in_scene_lights": [fixture_component] if fixture_component else [],
            "camera": {
                "name": camera.name,
                "strategy": "scene_free_space_v2",
                "location": [float(value) for value in camera.location],
                "rotation_euler": [float(value) for value in camera.rotation_euler],
                "focal_length": float(camera.data.lens),
                "target": [float(value) for value in anchor],
                "coordinate_space": "blender_world_meter",
                "yaw_degrees": float(job["camera"]["yaw_degrees"]),
                "pitch_degrees": float(job["camera"]["pitch_degrees"]),
                "shift_x": float(camera.data.shift_x),
                "shift_y": float(camera.data.shift_y),
                "sensor_width": float(camera.data.sensor_width),
                "sensor_fit": str(camera.data.sensor_fit),
                "angle_x": float(camera.data.angle_x),
                "angle_y": float(camera.data.angle_y),
                "clip_start": float(camera.data.clip_start),
                "resolution": {
                    "width": int(bpy.context.scene.render.resolution_x),
                    "height": int(bpy.context.scene.render.resolution_y),
                },
                "intrinsics": _camera_intrinsics(camera),
            },
            "camera_gate": gate,
            "canonical": {
                "origin": [float(value) for value in anchor],
                "asset_size": anchor_depth,
                "position_axes": "x=right,y=camera-forward,z=up",
                "units": "meter",
            },
            "lighting_profile": job["lighting_profile"],
            "lineage": job["lineage"],
            "license": job.get("license", {}),
        }
        if passes_metadata is not None:
            metadata["passes"] = passes_metadata
        (partial / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final_directory)
        return (final_directory / "metadata.json").relative_to(output_root).as_posix()
    except BaseException:
        if partial.exists():
            (partial / "failure.json").write_text(
                json.dumps({"job_id": job.get("job_id"), "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
        raise


# --------------------------------------------------------------------------- #
# Camera: exact pose validation, then build the Blender camera
# --------------------------------------------------------------------------- #
def _camera_basis(job: dict[str, Any]) -> tuple[Vector, Vector, Vector]:
    camera = job["camera"]
    forward = Vector(tuple(float(value) for value in camera["forward"])).normalized()
    right = Vector(tuple(float(value) for value in camera["right"])).normalized()
    up = Vector(tuple(float(value) for value in camera["up"])).normalized()
    return forward, right, up


def _validate_camera_pose(job: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Re-check the planned pose against real triangles.

    The planner only sees node AABBs, so this is the authoritative gate: a pose
    that would clip through geometry, stare at a single surface, or leave most of
    the frame empty fails the job before any EXR is written.
    """
    config = runtime.get("camera_gate", {})
    location = Vector(tuple(float(value) for value in job["camera"]["location"]))
    forward, right, up = _camera_basis(job)
    angle_x = float(job["camera"]["angle_x"])
    angle_y = float(job["camera"]["angle_y"])

    camera_radius = float(config.get("camera_radius", 0.20))
    near_clearance = float(config.get("near_clearance", 0.30))
    grid = max(3, int(config.get("ray_grid", 16)))
    max_distance = float(config.get("max_depth", 45.0))
    minimum_hit_fraction = float(config.get("min_hit_fraction", 0.90))
    max_near_fraction = float(config.get("max_near_fraction", 0.25))
    near_window = float(config.get("near_window", 1.00))
    max_single_object = float(config.get("max_single_object_fraction", 0.90))
    min_depth_spread = float(config.get("min_depth_spread_ratio", 1.35))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    clearance = _camera_clearance(location, camera_radius * 4.0)
    if clearance < camera_radius:
        raise ValueError(
            f"scene camera rejected: camera_inside_geometry (clearance={clearance:.3f} < {camera_radius})"
        )

    tan_x = math.tan(angle_x * 0.5) * 1.02
    tan_y = math.tan(angle_y * 0.5) * 1.02
    depths: list[float] = []
    hits_by_object: dict[str, int] = {}
    misses = 0
    for row in range(grid):
        offset_y = (2.0 * (row + 0.5) / grid) - 1.0
        for column in range(grid):
            offset_x = (2.0 * (column + 0.5) / grid) - 1.0
            direction = (forward + right * (offset_x * tan_x) + up * (offset_y * tan_y)).normalized()
            hit, hit_location, _normal, _face, obj, _matrix = bpy.context.scene.ray_cast(
                depsgraph, location, direction, distance=max_distance
            )
            if not hit or obj is None:
                misses += 1
                continue
            depth = (hit_location - location).length
            depths.append(depth)
            hits_by_object[obj.name] = hits_by_object.get(obj.name, 0) + 1

    total = grid * grid
    hit_fraction = len(depths) / total
    if hit_fraction < minimum_hit_fraction:
        raise ValueError(
            f"scene camera rejected: no_geometry_in_view (hit_fraction={hit_fraction:.3f} < {minimum_hit_fraction})"
        )
    minimum = min(depths)
    if minimum < near_clearance:
        raise ValueError(
            f"scene camera rejected: near_geometry_in_view (min_depth={minimum:.3f} < {near_clearance})"
        )
    near_fraction = sum(1 for depth in depths if depth < near_window) / len(depths)
    if near_fraction > max_near_fraction:
        raise ValueError(
            f"scene camera rejected: closeup_single_surface (near_fraction={near_fraction:.3f} > {max_near_fraction})"
        )
    dominant_name, dominant_hits = max(hits_by_object.items(), key=lambda item: item[1])
    dominance = dominant_hits / len(depths)
    if dominance > max_single_object:
        raise ValueError(
            "scene camera rejected: single_object_dominates "
            f"({dominant_name} covers {dominance:.3f} > {max_single_object})"
        )
    ordered = sorted(depths)
    low = ordered[max(0, int(len(ordered) * 0.10) - 1)]
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    spread = high / max(low, 1e-6)
    if spread < min_depth_spread:
        raise ValueError(
            f"scene camera rejected: depth_spread_too_small (spread={spread:.3f} < {min_depth_spread})"
        )
    return {
        "camera_clearance": clearance,
        "hit_fraction": hit_fraction,
        "hit_rays": len(depths),
        "miss_rays": misses,
        "ray_count": total,
        "minimum_hit_depth": minimum,
        "near_fraction": near_fraction,
        "single_object_fraction": dominance,
        "dominant_object": dominant_name,
        "depth_p10": low,
        "depth_p90": high,
        "depth_spread_ratio": spread,
    }


def _camera_clearance(location: Vector, radius: float) -> float:
    """Distance from the camera centre to the nearest rendered surface."""
    best = radius
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        if obj.type != "MESH" or obj.hide_render:
            continue
        minimum, maximum = _world_bounds([obj])
        if _point_aabb_distance(location, minimum, maximum) > best:
            continue
        bvh = _world_bvh(obj)
        if bvh is None:
            continue
        nearest = bvh.find_nearest(location)
        if nearest is not None:
            best = min(best, float(nearest[3]))
    return best


def _apply_camera(job: dict[str, Any], render_config: dict[str, Any], gate: dict[str, Any]):
    camera_config = job["camera"]
    data = bpy.data.cameras.new("LC_SceneCamera")
    camera = bpy.data.objects.new("LC_SceneCamera", data)
    bpy.context.collection.objects.link(camera)
    data.lens = float(camera_config["focal_length"])
    data.sensor_width = float(camera_config.get("sensor_width", 36.0))
    data.sensor_fit = str(camera_config.get("sensor_fit", "AUTO"))
    data.shift_x = float(camera_config.get("shift_x", 0.0))
    data.shift_y = float(camera_config.get("shift_y", 0.0))
    near = float(gate["minimum_hit_depth"])
    data.clip_start = max(0.01, min(near * 0.5, near - 1e-3))
    camera.location = Vector(tuple(float(value) for value in camera_config["location"]))
    rotation = camera_config.get("rotation_euler")
    if rotation is None:
        raise ValueError("scene camera job has no rotation_euler")
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = tuple(float(value) for value in rotation)
    resolution = camera_config.get("resolution") or {}
    width = int(resolution.get("width") or render_config["resolution"])
    height = int(resolution.get("height") or render_config["resolution"])
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    return camera


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #
def _select_native_fixture(job, runtime, camera, anchor, partial, helpers, rng):
    """Pick a visible scene-native fixture (lamp sconce...) or fall back to one."""
    config = runtime.get("fixture", {})
    resolution = int(_render_config(runtime.get("render", {}))["resolution"])
    maximum_screen_fraction = float(config.get("maximum_screen_fraction", 0.05))
    minimum_pixels = int(config.get("minimum_visible_pixels", 64))
    candidates: list[dict[str, Any]] = []
    for entity_id in job.get("fixture_entity_ids", []):
        objects = [
            obj for obj in _entity_objects(entity_id) if obj.type == "MESH" and not obj.hide_render
        ]
        if not objects:
            continue
        mask_path = partial / "diagnostics" / f"fixture_candidate_{_safe_name(entity_id)}.png"
        pixels = _render_mask(mask_path, objects, _render_config(runtime.get("render", {})))
        if pixels < minimum_pixels:
            continue
        screen_fraction = pixels / float(resolution**2)
        if screen_fraction > maximum_screen_fraction:
            continue
        center = sum((_object_center(obj) for obj in objects), Vector()) / len(objects)
        ndc = world_to_camera_view(bpy.context.scene, camera, center)
        candidates.append(
            {
                "source": "scene_native",
                "entity_id": entity_id,
                "objects": objects,
                "world_position": center,
                "visible_pixels": pixels,
                "screen_fraction": screen_fraction,
                "center_distance": math.hypot(float(ndc.x) - 0.5, float(ndc.y) - 0.5),
                "candidate_mask": mask_path,
                "material": None,
            }
        )
    if candidates:
        candidates.sort(key=lambda row: (row["center_distance"], row["entity_id"]))
        return candidates[0]

    # No usable native lamp in view: place an artificial emitter in front of the
    # camera at a depth proportional to the framed region, so it lights the
    # visible content instead of the void behind the camera.
    size = max(
        float(config.get("fallback_fixture_size_min", 0.02)),
        min(
            float(config.get("fallback_fixture_size_max", 0.25)),
            float(job["canonical"]["asset_size"]) * float(config.get("fallback_fixture_size_ratio", 0.10)),
        ),
    )
    ranges = config.get(
        "fallback_position_ranges",
        {"x": [-0.4, 0.4], "y": [0.3, 0.8], "z": [-0.3, 0.3]},
    )
    depth = float(job["canonical"]["asset_size"])
    for candidate_index in range(int(config.get("fallback_candidates", 24))):
        offset = [rng.uniform(*ranges[axis]) * depth for axis in ("x", "y", "z")]
        world_position = Vector(tuple(float(value) for value in anchor)) + Vector(offset)
        mesh, material = helpers.add_fixture(
            f"LC_SceneFixture_{candidate_index:03d}", world_position, size
        )
        mask_path = partial / "diagnostics" / f"fixture_fallback_{candidate_index:03d}.png"
        pixels = _render_mask(mask_path, [mesh], _render_config(runtime.get("render", {})))
        if pixels >= minimum_pixels:
            return {
                "source": "procedural_fallback",
                "entity_id": None,
                "objects": [mesh],
                "world_position": world_position,
                "visible_pixels": pixels,
                "screen_fraction": pixels / float(resolution**2),
                "candidate_mask": mask_path,
                "material": material,
            }
        _remove_fixture_mesh(mesh, material)
    raise ValueError("no fixture could be selected for this camera")


def _render_fixture_component(
    fixture, partial, output_root, final_directory, config, helpers, anchor, passes_ctx=None
):
    mask_path = partial / "in_scene_lights" / "fixture_000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture["candidate_mask"], mask_path)
    energy = float(config["fixture_energy"])
    radius = float(config["fixture_light_radius"])
    light = helpers.add_light("LC_FixtureLight", "POINT", fixture["world_position"], energy, radius)
    if fixture.get("material") is not None:
        helpers.set_fixture_emission(
            fixture["material"],
            (1.0, 1.0, 1.0, 1.0),
            float(config["fallback_emission_strength"]),
        )
    try:
        component_path = partial / "in_scene_lights" / "fixture_000_on.exr"
        if passes_ctx is not None and bool(passes_ctx["config"].get("irradiance", True)):
            outcome = helpers.render_exr(
                component_path,
                passes_ctx["dir"],
                ("irradiance",),
                name="fixture_000",
                eps=passes_ctx["eps"],
            )
            if "irradiance" in outcome:
                passes_ctx["irradiance"]["fixture_000"] = _future_relative(
                    outcome["irradiance"], partial, final_directory, output_root
                )
        else:
            helpers.render_exr(component_path)
    finally:
        if fixture.get("material") is not None:
            helpers.set_fixture_emission(fixture["material"], (1.0, 1.0, 1.0, 1.0), 0.0)
        helpers.remove_light(light)
    return {
        "path": _future_relative(component_path, partial, final_directory, output_root),
        "mask": _future_relative(mask_path, partial, final_directory, output_root),
        "position": _world_to_canonical(fixture["world_position"], anchor),
        "renderer_position": [float(value) for value in fixture["world_position"]],
        "base_energy": energy,
        "diffuse": radius,
        "fixture_source": fixture["source"],
        "fixture_entity_id": fixture["entity_id"],
        "visible_pixels": int(fixture["visible_pixels"]),
        "screen_fraction": float(fixture["screen_fraction"]),
        "binding": "controlled_point_at_fixture_center",
    }


def _world_to_canonical(world_position, anchor) -> list[float]:
    """Canonical ``[right, forward, up]`` offset from the camera target.

    The canonical frame is the renderer's ``x=right, y=camera-forward, z=up``
    with the origin at ``camera.target``; ``coordinates.py`` re-derives it from
    the metadata camera block, so this must be world-consistent.
    """
    target = Vector(tuple(float(value) for value in anchor))
    location = Vector(tuple(_canonical_camera_location()))
    forward = (target - location).normalized()
    right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
    up = right.cross(forward).normalized()
    delta = Vector(world_position) - target
    return [float(delta.dot(right)), float(delta.dot(forward)), float(delta.dot(up))]


def _canonical_camera_location() -> list[float]:
    camera = bpy.context.scene.camera
    if camera is None:
        raise ValueError("scene has no active camera")
    return [float(value) for value in camera.location]


def _render_point_components(
    partial, output_root, final_directory, camera, target, anchor_depth, config, helpers, rng, passes_ctx=None
):
    count = int(config["point_lights_per_scene"])
    if count < 1:
        return []
    light = helpers.add_light("LC_PointLight", "POINT", (0, 0, 0), float(config["point_energy"]), 0.0)
    rows = []
    try:
        for index in range(count):
            # Ranges are fractions of the framed depth: a fixed metre offset would
            # put every light inside the camera in a large scene.
            position = [
                rng.uniform(*config["point_position_ranges"][axis]) * anchor_depth
                for axis in ("x", "y", "z")
            ]
            world_position = helpers.camera_to_world(camera, target, position)
            radius = rng.uniform(*config["point_radius_range"])
            light.location = world_position
            light.data.shadow_soft_size = radius
            bpy.context.view_layer.update()
            path = partial / "point_lights" / f"light_{index:03d}.exr"
            label = f"point_{index:03d}"
            if passes_ctx is not None and bool(passes_ctx["config"].get("irradiance", True)):
                outcome = helpers.render_exr(
                    path, passes_ctx["dir"], ("irradiance",), name=label, eps=passes_ctx["eps"]
                )
                if "irradiance" in outcome:
                    passes_ctx["irradiance"][label] = _future_relative(
                        outcome["irradiance"], partial, final_directory, output_root
                    )
            else:
                helpers.render_exr(path)
            rows.append(
                {
                    "path": _future_relative(path, partial, final_directory, output_root),
                    "position": position,
                    "renderer_position": [float(value) for value in world_position],
                    "base_energy": float(config["point_energy"]),
                    "diffuse": radius,
                }
            )
    finally:
        helpers.remove_light(light)
    return rows


def _render_diffuse_components(
    partial, output_root, final_directory, camera, target, anchor_depth, config, helpers
):
    spreads = [float(value) for value in config["diffuse_sizes"]]
    if len(spreads) < 2:
        raise ValueError("at least two diffuse sizes are required")
    position = [float(value) * anchor_depth for value in config["diffuse_position"]]
    world_position = helpers.camera_to_world(camera, target, position)
    light = helpers.add_light(
        "LC_DiffuseLight", "AREA", world_position, float(config["point_energy"]), spreads[0]
    )
    _look_at(light, target)
    rows = []
    try:
        for index, spread in enumerate(spreads):
            light.data.size = spread
            bpy.context.view_layer.update()
            path = partial / "diffuse" / f"spread_{index:02d}.exr"
            helpers.render_exr(path)
            rows.append(
                {
                    "path": _future_relative(path, partial, final_directory, output_root),
                    "level": index / (len(spreads) - 1),
                    "size": spread,
                }
            )
    finally:
        helpers.remove_light(light)
    return rows


def _render_mask(path: Path, selected_objects, config: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    original_engine = scene.render.engine
    original_path = scene.render.filepath
    image_settings = scene.render.image_settings
    original_format = image_settings.file_format
    original_mode = image_settings.color_mode
    original_depth = image_settings.color_depth
    original_film = scene.render.film_transparent
    colors = {obj: tuple(obj.color) for obj in bpy.data.objects if obj.type == "MESH"}
    hidden = {obj: bool(obj.hide_render) for obj in colors}
    selected = set(selected_objects)
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "OBJECT"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = False
        scene.display.shading.show_specular_highlight = False
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = (0.0, 0.0, 0.0)
        for obj in colors:
            obj.color = (1.0, 1.0, 1.0, 1.0) if obj in selected else (0.0, 0.0, 0.0, 1.0)
        image_settings.file_format = "PNG"
        image_settings.color_mode = "BW"
        image_settings.color_depth = "8"
        scene.render.film_transparent = False
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, color in colors.items():
            obj.color = color
            obj.hide_render = hidden[obj]
        scene.render.engine = original_engine
        scene.render.filepath = original_path
        image_settings.file_format = original_format
        image_settings.color_mode = original_mode
        image_settings.color_depth = original_depth
        scene.render.film_transparent = original_film
        _apply_cycles_settings(scene, config)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = list(image.pixels)
        return sum(pixels[index] > 0.5 for index in range(0, len(pixels), 4))
    finally:
        bpy.data.images.remove(image)


def _disable_native_lighting() -> None:
    """Turn off every light the source scene brought, so components stay separable."""
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.data.energy = 0.0
    for material in bpy.data.materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                strength = node.inputs.get("Emission Strength")
                color = node.inputs.get("Emission Color")
                if color is None:
                    color = node.inputs.get("Emission")
                if strength is not None:
                    strength.default_value = 0.0
                if color is not None:
                    color.default_value = (0.0, 0.0, 0.0, 1.0)
            elif node.type == "EMISSION" and node.inputs.get("Strength") is not None:
                node.inputs["Strength"].default_value = 0.0


def _configure_ambient_world(config: dict[str, Any], job: dict[str, Any]):
    world = bpy.data.worlds.new("LC_SceneWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    hdri = _select_ambient_hdri(config.get("hdri_root"), job)
    if hdri is not None:
        environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(str(hdri), check_existing=True)
        world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
        print(f"SCENE_HDRI {job['job_id']}: {hdri}", flush=True)
    else:
        background.inputs["Color"].default_value = tuple(
            float(value) for value in config["ambient_world_color"]
        )
    background.inputs["Strength"].default_value = float(config["ambient_world_strength"])
    return background, hdri


def _select_ambient_hdri(root_value: Any, job: dict[str, Any]) -> Path | None:
    if root_value in (None, ""):
        return None
    root = Path(str(root_value)).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"HDRI root is not a directory: {root}")
    candidates = _ambient_hdri_candidates(str(root))
    if not candidates:
        raise ValueError(f"HDRI root contains no .hdr or .exr files: {root}")
    selector = hashlib.sha256(
        f"{job['seed']}\0{job['job_id']}\0scene-ambient-v2".encode("utf-8")
    ).digest()
    return candidates[int.from_bytes(selector[:8], "big") % len(candidates)]


@lru_cache(maxsize=8)
def _ambient_hdri_candidates(root_value: str) -> tuple[Path, ...]:
    root = Path(root_value)
    return tuple(
        sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".hdr", ".exr"}
        )
    )


# --------------------------------------------------------------------------- #
# Render configuration and small helpers
# --------------------------------------------------------------------------- #
def _render_config(value: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "resolution": 512,
        "engine": "CYCLES",
        "samples": 64,
        "denoise": True,
        "persistent_data": False,
        "threads_per_worker": "auto",
        "device": "GPU",
        "compute_device_type": "CUDA",
        "require_gpu": True,
        "hdri_root": None,
        "ambient_world_strength": 1.0,
        "ambient_world_color": [0.18, 0.18, 0.18, 1.0],
        "point_lights_per_scene": 16,
        "point_energy": 500.0,
        "point_position_ranges": {"x": [-0.6, 0.6], "y": [0.25, 0.75], "z": [-0.4, 0.4]},
        "point_radius_range": [0.03, 0.25],
        "diffuse_sizes": [0.05, 0.2, 0.4, 0.7, 1.0, 1.5],
        "diffuse_position": [0.0, 0.25, 0.35],
        "fixture_energy": 300.0,
        "fixture_light_radius": 0.12,
        "fallback_emission_strength": 4.0,
        "passes": {
            "enabled": False,
            "albedo": True,
            "normal": True,
            "irradiance": True,
            "irradiance_mode": "derived",
            "normal_space": "camera",
            "save_normal_world": False,
            "pass_samples": 1,
            "eps": 0.001,
            "denoise_passes": False,
            "require_textures": False,
            "min_base_color_linked_ratio": 0.9,
            "max_magenta_ratio": 0.0001,
        },
    }
    result = {**defaults, **(value or {})}
    result["passes"] = {**defaults["passes"], **((value or {}).get("passes") or {})}
    return result


def _apply_cycles_settings(scene, config: dict[str, Any]) -> None:
    scene.render.engine = "BLENDER_EEVEE_NEXT" if config["engine"] == "EEVEE" else "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(config["samples"])
        scene.cycles.use_denoising = bool(config["denoise"])


def _scene_fingerprint() -> str:
    rows = []
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "matrix": _matrix_flat(obj.matrix_world),
                "hide_render": bool(obj.hide_render),
                "entity_id": obj.get("lc_scene_entity_id"),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _entity_objects(entity_id: str):
    return sorted(
        (obj for obj in bpy.data.objects if obj.get("lc_scene_entity_id") == entity_id),
        key=lambda obj: obj.name,
    )


def _world_bounds(objects):
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise ValueError("cannot compute bounds for an empty mesh set")
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def _world_bvh(obj, *, epsilon: float = 0.0) -> BVHTree | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh.vertices or not mesh.polygons:
            return None
        matrix = evaluated.matrix_world
        vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        polygons = [tuple(polygon.vertices) for polygon in mesh.polygons]
        return BVHTree.FromPolygons(vertices, polygons, all_triangles=False, epsilon=epsilon)
    finally:
        evaluated.to_mesh_clear()


def _point_aabb_distance(point: Vector, minimum: Vector, maximum: Vector) -> float:
    squared = 0.0
    for axis in range(3):
        delta = max(float(minimum[axis]) - point[axis], 0.0, point[axis] - float(maximum[axis]))
        squared += delta * delta
    return math.sqrt(squared)


def _object_center(obj) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def _remove_fixture_mesh(mesh, material) -> None:
    mesh_data = mesh.data
    bpy.data.objects.remove(mesh, do_unlink=True)
    if mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)
    if material is not None and material.users == 0:
        bpy.data.materials.remove(material)


def _camera_intrinsics(camera) -> dict[str, float | int]:
    width = int(bpy.context.scene.render.resolution_x)
    height = int(bpy.context.scene.render.resolution_y)
    fx = (width * 0.5) / math.tan(float(camera.data.angle_x) * 0.5)
    fy = (height * 0.5) / math.tan(float(camera.data.angle_y) * 0.5)
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5 - width * float(camera.data.shift_x),
        "cy": height * 0.5 + height * float(camera.data.shift_y),
    }


def _look_at(obj, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _matrix_flat(matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _future_relative(path: Path, partial: Path, final: Path, output_root: Path) -> str:
    future = final / path.relative_to(partial)
    return future.relative_to(output_root).as_posix()


def _safe_name(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_reusable_metadata(metadata_path: Path, job: dict[str, Any]) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot reuse invalid metadata: {metadata_path}") from error
    expected = str(job.get("lineage", {}).get("scene_camera_digest") or "")
    actual = str(metadata.get("lineage", {}).get("scene_camera_digest") or "")
    if metadata.get("id") != job["job_id"] or not expected or actual != expected:
        raise FileExistsError(
            "completed output does not match the current render job; use a new output root "
            f"or explicitly archive this stale job directory first: {metadata_path.parent}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _datacompose_root() -> Path:
    # Blender runs this file by path, so resolve sibling modules from __file__
    # instead of trusting a caller-provided project root.
    return Path(__file__).resolve().parents[1]


def _load_helpers():
    path = _datacompose_root() / "tokenlight" / "render_assets.py"
    spec = importlib.util.spec_from_file_location("tokenlight_render_assets", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_recovery():
    path = _datacompose_root() / "render_recovery.py"
    spec = importlib.util.spec_from_file_location("datacompose_render_recovery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load render recovery helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_intrinsic():
    path = _datacompose_root() / "intrinsic.py"
    spec = importlib.util.spec_from_file_location("datacompose_intrinsic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load intrinsic helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_coordinates():
    path = _datacompose_root() / "coordinates.py"
    spec = importlib.util.spec_from_file_location("datacompose_coordinates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load coordinate helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    # coordinates.py defines frozen dataclasses and dataclasses resolves the class
    # module through sys.modules, so register it before exec_module runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


if __name__ == "__main__":
    main()
