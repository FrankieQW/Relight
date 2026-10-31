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
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    runtime = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    project_root = Path(runtime["project_root"])
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
            print(f"COMPOSITION_DONE {job['job_id']}", flush=True)
        except Exception as error:  # noqa: BLE001 - every failed render remains recoverable by job id.
            errors.append(
                {
                    "job_id": job.get("job_id"),
                    "error": type(error).__name__,
                    "detail": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"COMPOSITION_FAILED {job.get('job_id')}: {error}", flush=True)
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
    object_root = Path(runtime["object_root"]).resolve()
    object_path = (object_root / job["object_asset_path"]).resolve()
    try:
        object_path.relative_to(object_root)
    except ValueError as error:
        raise ValueError("object_asset_path resolves outside OBJECT_ROOT") from error
    if not blend_path.is_file():
        raise FileNotFoundError(blend_path)
    if not object_path.is_file():
        raise FileNotFoundError(object_path)
    if _sha256_file(blend_path) != job["base_scene_digest"]:
        raise ValueError("base scene blend digest changed after render-job generation")
    if _sha256_file(object_path) != job["prepared_geometry"]["asset_digest"]:
        raise ValueError("object asset digest changed after render-job generation")
    if job.get("license", {}).get("decision") != "allowed" and bool(
        runtime.get("render", {}).get("require_verified_license", True)
    ):
        raise ValueError("render job license decision is not allowed")
    # Do not create a partial directory until all preflight checks pass.  A
    # preflight error has no render state to recover and must not leave an
    # unclearable partial without failure.json.
    partial = recovery.prepare_render_partial(output_root, job["job_id"])

    try:
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        render_config = _render_config(runtime.get("render", {}))
        helpers.configure_render(render_config)
        base_fingerprint = _scene_fingerprint()
        target_objects = _entity_objects(job["target"]["entity_id"])
        if not target_objects:
            raise ValueError(f"target entity not found in blend: {job['target']['entity_id']}")
        hidden_target_names: list[str] = []
        if job["target"]["relation"] == "replace":
            for obj in target_objects:
                obj.hide_render = True
                obj.hide_viewport = True
                hidden_target_names.append(obj.name)

        imported, meshes = helpers.import_asset(object_path)
        root, initial_scale = _normalize_imported(
            imported, meshes, job["prepared_geometry"], job["target"]
        )
        placement = _resolve_inserted_placement(
            root,
            meshes,
            target_objects,
            job["target"],
            render_config,
        )
        final_scale = placement["final_scale"]
        collision_pairs = placement["collision_pairs"]
        # The first camera is provisional: it provides a view for choosing a
        # visible fixture.  Once the fixture exists, the final camera search
        # runs again with that geometry participating in clearance, occlusion,
        # and screen-coverage gates.
        provisional_camera, camera_target, provisional_result = _select_camera(
            job,
            target_objects,
            meshes,
            partial,
            render_config,
        )
        rng = random.Random(int(job["seed"]))
        fixture = _select_or_create_fixture(
            job,
            runtime,
            provisional_camera,
            camera_target,
            provisional_result["subject_diameter"],
            partial,
            helpers,
            rng,
        )
        _remove_camera(provisional_camera)
        camera, camera_target, camera_result = _select_camera(
            job,
            target_objects,
            meshes,
            partial,
            render_config,
            fixture=fixture,
            fixture_config=runtime.get("fixture", {}),
        )
        bpy.context.scene.camera = camera
        fixture["candidate_mask"] = camera_result["fixture_mask_path"]
        fixture["visible_pixels"] = int(camera_result["fixture_visible_pixels"])
        fixture["screen_fraction"] = float(camera_result["fixture_screen_fraction"])
        fixture["position"] = _world_to_canonical(
            camera, camera_target, fixture["world_position"]
        )
        _disable_native_lighting()
        background, ambient_hdri = _configure_ambient_world(render_config, job)

        visibility_path = partial / "diagnostics" / "inserted_visibility.png"
        shutil.copyfile(camera_result["mask_path"], visibility_path)
        inserted_pixels = int(camera_result["visible_pixels"])
        ambient_path = partial / "ambient.exr"
        background.inputs["Strength"].default_value = float(render_config["ambient_world_strength"])
        helpers.render_exr(ambient_path)
        background.inputs["Strength"].default_value = 0.0
        dark_path = partial / "dark.exr"
        helpers.render_exr(dark_path)

        fixture_component = _render_fixture_component(
            fixture, partial, output_root, final_directory, render_config, helpers
        )
        point_components = _render_point_components(
            partial, output_root, final_directory, camera, camera_target, render_config, helpers, rng
        )
        diffuse_components = _render_diffuse_components(
            partial, output_root, final_directory, camera, camera_target, render_config, helpers
        )
        canonical = {
            "origin": [float(value) for value in camera_target],
            "asset_size": float(camera_result["subject_diameter"]),
            "position_axes": "x=right,y=camera-forward,z=up",
            "units": "meter",
        }
        metadata = {
            "schema_version": job["schema_version"],
            "id": job["job_id"],
            "asset_uid": job["object_uid"],
            "asset": job["object_asset_path"],
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
            "in_scene_lights": [fixture_component],
            "camera": {
                "name": camera.name,
                "location": [float(value) for value in camera.location],
                "rotation_euler": [float(value) for value in camera.rotation_euler],
                "focal_length": float(camera.data.lens),
                "target": [float(value) for value in camera_target],
                "coordinate_space": "blender_world_meter",
                "strategy": "generated_target_visible",
                "candidate_index": camera_result["candidate_index"],
                "coarse_candidate_index": camera_result["coarse_candidate_index"],
                "repair_index": camera_result["repair_index"],
                "azimuth_degrees": camera_result["azimuth_degrees"],
                "elevation_degrees": camera_result["elevation_degrees"],
                "shift_x": float(camera.data.shift_x),
                "shift_y": float(camera.data.shift_y),
                # Pinhole parameters let downstream code project lights onto the
                # image plane without re-deriving Blender's sensor model.
                "sensor_width": float(camera.data.sensor_width),
                "sensor_fit": str(camera.data.sensor_fit),
                "angle_x": float(camera.data.angle_x),
                "angle_y": float(camera.data.angle_y),
                "resolution": {
                    "width": int(bpy.context.scene.render.resolution_x),
                    "height": int(bpy.context.scene.render.resolution_y),
                },
                "intrinsics": _camera_intrinsics(camera),
                "inserted_ndc_bounds": camera_result["inserted_ndc_bounds"],
                "target_entity_ndc": camera_result["target_entity_ndc"],
                "target_visible_pixels": camera_result["target_visible_pixels"],
                "inserted_reference_pixels": camera_result["reference_pixels"],
                "inserted_visible_ratio": camera_result["visible_ratio"],
                "clearance": camera_result["camera_clearance"],
                "near_foreground_fraction": camera_result["near_foreground_fraction"],
            },
            "canonical": canonical,
            "composition": {
                "relation": job["target"]["relation"],
                "target_entity_id": job["target"]["entity_id"],
                "target_node_ids": job["target"]["node_ids"],
                "object_transform_world": _matrix_flat(root.matrix_world),
                "target_transform_before": [
                    {"name": obj.name, "matrix_world": _matrix_flat(obj.matrix_world)}
                    for obj in target_objects
                ],
                "hidden_target_objects": hidden_target_names,
                "asset_scale": final_scale,
                "initial_asset_scale": [float(value) for value in initial_scale],
                "inserted_visible_pixels": inserted_pixels,
                "collision_pairs": collision_pairs,
                "collision_resolution": placement["collision_resolution"],
            },
            "lighting_profile": job["lighting_profile"],
            "base_scene_id": job["base_scene_id"],
            "base_scene_fingerprint": base_fingerprint,
            "lineage": job["lineage"],
            "license": job["license"],
        }
        (partial / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final_directory)
        return (final_directory / "metadata.json").relative_to(output_root).as_posix()
    except BaseException:
        if partial.exists():
            failure = partial / "failure.json"
            failure.write_text(
                json.dumps({"job_id": job.get("job_id"), "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
        raise


def _validate_reusable_metadata(metadata_path: Path, job: dict[str, Any]) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot reuse invalid metadata: {metadata_path}") from error
    expected_digest = str(job.get("lineage", {}).get("render_job_digest") or "")
    actual_digest = str(metadata.get("lineage", {}).get("render_job_digest") or "")
    if metadata.get("id") != job.get("job_id") or not expected_digest or actual_digest != expected_digest:
        raise FileExistsError(
            "completed output does not match the current render job; use a new output root "
            f"or explicitly archive this stale job directory first: {metadata_path.parent}"
        )


def _remove_camera(camera) -> None:
    data = camera.data
    bpy.data.objects.remove(camera, do_unlink=True)
    if data.users == 0:
        bpy.data.cameras.remove(data)


def _normalize_imported(imported, meshes, geometry: dict[str, Any], target: dict[str, Any]):
    root = bpy.data.objects.new(f"LC_ObjectRoot_{geometry['object_uid']}", None)
    bpy.context.collection.objects.link(root)
    for obj in imported:
        if obj.parent is None:
            matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = matrix
    source_up = _axis_vector(geometry["up_axis"])
    source_front = _axis_vector(geometry["front_axis"])
    up_rotation = source_up.rotation_difference(Vector((0.0, 0.0, 1.0)))
    rotated_front = up_rotation @ source_front
    rotated_front.z = 0.0
    if rotated_front.length <= 1e-8:
        raise ValueError("front axis collapses after up-axis normalization")
    rotated_front.normalize()
    target_front = Vector((0.0, -1.0, 0.0))
    signed_angle = math.atan2(rotated_front.cross(target_front).z, rotated_front.dot(target_front))
    yaw = math.radians(float(target["yaw_degrees"]))
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), yaw + signed_angle) @ up_rotation
    bpy.context.view_layer.update()
    minimum, maximum = _world_bounds(meshes)
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    if any(value <= 1e-8 for value in dimensions):
        raise ValueError(f"imported asset has a degenerate bounding box: {dimensions}")
    desired = [float(value) for value in target["desired_dimensions"]]
    if geometry["fit_mode"] == "uniform_fit":
        scalar = min(desired[index] / dimensions[index] for index in range(3))
        scale = [scalar, scalar, scalar]
    else:
        scale = [desired[index] / dimensions[index] for index in range(3)]
    root.scale = scale
    bpy.context.view_layer.update()
    minimum, maximum = _world_bounds(meshes)
    bottom_center = Vector(
        ((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, minimum.z)
    )
    destination = Vector(target["bottom_center_world"])
    root.location += destination - bottom_center
    bpy.context.view_layer.update()
    return root, scale


def _place_footprint_status(
    meshes, target: dict[str, Any], render: dict[str, Any]
) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    minimum, maximum = _world_bounds(meshes)
    object_xy = (maximum.x - minimum.x, maximum.y - minimum.y)
    target_xy = tuple(float(value) for value in target["dimensions_world"][:2])
    limit = float(render.get("place_footprint_limit", 1.0))
    fits = all(object_xy[index] <= target_xy[index] * limit for index in range(2))
    return fits, object_xy, target_xy


def _align_bottom_to_target(root, meshes, target: dict[str, Any]) -> None:
    minimum, maximum = _world_bounds(meshes)
    bottom_center = Vector(
        ((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, minimum.z)
    )
    root.location += Vector(target["bottom_center_world"]) - bottom_center
    bpy.context.view_layer.update()


def _resolve_inserted_placement(
    root,
    meshes,
    target_objects,
    target: dict[str, Any],
    render: dict[str, Any],
) -> dict[str, Any]:
    """Keep the largest deterministic scale that satisfies placement constraints.

    Every retry is a uniform multiplier of the initially normalized scale.  The
    object is re-anchored at the target bottom center after scaling so shrinking
    cannot make it float above or drift across the support surface.
    """
    enabled = bool(render.get("collision_shrink_enabled", True))
    shrink_factor = float(render.get("collision_shrink_factor", 0.9))
    minimum_ratio = float(render.get("collision_min_scale_ratio", 0.6))
    maximum_attempts = int(render.get("collision_max_attempts", 6))
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("render.collision_shrink_factor must be in (0, 1)")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("render.collision_min_scale_ratio must be in (0, 1]")
    if maximum_attempts < 1:
        raise ValueError("render.collision_max_attempts must be positive")

    ratios = [1.0]
    if enabled:
        while len(ratios) < maximum_attempts and ratios[-1] > minimum_ratio:
            next_ratio = max(minimum_ratio, ratios[-1] * shrink_factor)
            if math.isclose(next_ratio, ratios[-1], rel_tol=0.0, abs_tol=1e-12):
                break
            ratios.append(next_ratio)

    initial_scale = [float(value) for value in root.scale]
    scene_meshes = _collision_scene_meshes(meshes, target_objects)
    scene_bvh_cache: dict[Any, BVHTree | None] = {}
    scene_bounds_cache: dict[Any, tuple[Any, Any]] = {}
    initial_collision_pairs: list[list[str]] = []
    last_collision_pairs: list[list[str]] = []
    last_footprint: tuple[float, float] | None = None
    target_footprint: tuple[float, float] | None = None

    for attempt, ratio in enumerate(ratios, start=1):
        root.scale = [value * ratio for value in initial_scale]
        bpy.context.view_layer.update()
        _align_bottom_to_target(root, meshes, target)

        footprint_fits = True
        if target["relation"] == "place_on":
            footprint_fits, last_footprint, target_footprint = _place_footprint_status(
                meshes, target, render
            )
        last_collision_pairs = _find_scene_collisions(
            meshes,
            scene_meshes,
            render,
            scene_bvh_cache=scene_bvh_cache,
            scene_bounds_cache=scene_bounds_cache,
        )
        if attempt == 1:
            initial_collision_pairs = list(last_collision_pairs)
        if footprint_fits and not last_collision_pairs:
            final_scale = [float(value) for value in root.scale]
            return {
                "final_scale": final_scale,
                "collision_pairs": [],
                "collision_resolution": {
                    "strategy": "none" if attempt == 1 else "uniform_shrink",
                    "scale_ratio": float(ratio),
                    "attempts": attempt,
                    "initial_collision_count": len(initial_collision_pairs),
                    "initial_collision_pairs_preview": initial_collision_pairs[:5],
                    "footprint_adjusted": bool(attempt > 1 and target["relation"] == "place_on"),
                },
            }

    ratio = ratios[-1]
    if last_collision_pairs:
        preview = ", ".join(f"{left}<->{right}" for left, right in last_collision_pairs[:5])
        raise ValueError(
            "inserted object intersects non-target scene geometry after "
            f"{len(ratios)} scale attempt(s) down to ratio {ratio:.6g}: {preview}"
        )
    raise ValueError(
        f"placed object footprint {last_footprint} exceeds support {target_footprint} after "
        f"{len(ratios)} scale attempt(s) down to ratio {ratio:.6g}"
    )


def _collision_scene_meshes(meshes, target_objects) -> list[Any]:
    inserted = set(meshes)
    excluded = inserted | set(target_objects)
    return sorted(
        (
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH"
            and obj not in excluded
            and not obj.hide_render
            and not obj.hide_viewport
        ),
        key=lambda obj: obj.name,
    )


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
        return BVHTree.FromPolygons(
            vertices,
            polygons,
            all_triangles=False,
            epsilon=epsilon,
        )
    finally:
        evaluated.to_mesh_clear()


def _find_scene_collisions(
    meshes,
    scene_meshes,
    render: dict[str, Any],
    *,
    scene_bvh_cache: dict[Any, BVHTree | None] | None = None,
    scene_bounds_cache: dict[Any, tuple[Any, Any]] | None = None,
) -> list[list[str]]:
    """Use AABB only as broad phase and BVH triangle overlap as the verdict."""
    tolerance = float(render.get("collision_aabb_tolerance", 0.002))
    epsilon = float(render.get("collision_bvh_epsilon", 0.0))
    if tolerance < 0.0 or epsilon < 0.0:
        raise ValueError("collision tolerances must be non-negative")
    cache = {} if scene_bvh_cache is None else scene_bvh_cache
    bounds_cache = {} if scene_bounds_cache is None else scene_bounds_cache
    collisions: list[list[str]] = []
    for inserted_obj in sorted(set(meshes), key=lambda obj: obj.name):
        inserted_minimum, inserted_maximum = _world_bounds([inserted_obj])
        inserted_bvh = _world_bvh(inserted_obj, epsilon=epsilon)
        if inserted_bvh is None:
            continue
        for scene_obj in scene_meshes:
            if scene_obj not in bounds_cache:
                bounds_cache[scene_obj] = _world_bounds([scene_obj])
            scene_minimum, scene_maximum = bounds_cache[scene_obj]
            overlaps = [
                min(inserted_maximum[axis], scene_maximum[axis])
                - max(inserted_minimum[axis], scene_minimum[axis])
                for axis in range(3)
            ]
            if not all(value > tolerance for value in overlaps):
                continue
            if scene_obj not in cache:
                cache[scene_obj] = _world_bvh(scene_obj, epsilon=epsilon)
            scene_bvh = cache[scene_obj]
            if scene_bvh is not None and inserted_bvh.overlap(scene_bvh):
                collisions.append([inserted_obj.name, scene_obj.name])
    return collisions


def _validate_scene_collisions(meshes, target_objects, render: dict[str, Any]) -> list[list[str]]:
    collisions = _find_scene_collisions(
        meshes,
        _collision_scene_meshes(meshes, target_objects),
        render,
    )
    if collisions:
        preview = ", ".join(f"{left}<->{right}" for left, right in collisions[:5])
        raise ValueError(f"inserted object intersects non-target scene geometry: {preview}")
    return collisions


def _select_camera(
    job,
    target_objects,
    inserted_meshes,
    partial,
    render_config,
    *,
    fixture=None,
    fixture_config=None,
):
    """Create a deterministic camera around the composition and reject bad framing.

    Source-scene cameras are intentionally ignored: they may be arbitrarily far
    from the selected entity and are not part of the formal data contract.
    """
    config = job["camera"]
    if config.get("strategy") != "generated_target_visible":
        raise ValueError("formal composition jobs require generated_target_visible camera strategy")
    focal_length = float(config["focal_length"])
    if focal_length <= 0:
        raise ValueError("camera focal_length must be positive")
    azimuths = [float(value) for value in config.get("azimuth_degrees", [])]
    elevations = [float(value) for value in config.get("elevation_degrees", [])]
    if not azimuths or not elevations:
        raise ValueError("camera azimuth/elevation candidate lists cannot be empty")
    if any(value < 5.0 or value > 80.0 for value in elevations):
        raise ValueError("camera elevations must stay in [5, 80] degrees")
    fill_range = _ordered_pair(config.get("subject_fill_range"), "camera.subject_fill_range", 0.01, 0.95)
    shift_x_range = _ordered_pair(config.get("shift_x_range"), "camera.shift_x_range", -0.5, 0.5)
    shift_y_range = _ordered_pair(config.get("shift_y_range"), "camera.shift_y_range", -0.5, 0.5)
    ndc_x_range = _ordered_pair(config.get("ndc_x_range"), "camera.ndc_x_range", 0.0, 1.0)
    ndc_y_range = _ordered_pair(config.get("ndc_y_range"), "camera.ndc_y_range", 0.0, 1.0)
    edge_margin = float(config.get("edge_margin", 0.02))
    if not 0.0 <= edge_margin < 0.25:
        raise ValueError("camera edge_margin must be in [0, 0.25)")
    candidate_count = int(config.get("candidate_count", 24))
    if candidate_count < 1:
        raise ValueError("camera candidate_count must be positive")

    inserted_minimum, inserted_maximum = _world_bounds(inserted_meshes)
    anchor = (inserted_minimum + inserted_maximum) * 0.5
    dimensions = inserted_maximum - inserted_minimum
    subject_diameter = max(float(value) for value in dimensions)
    if subject_diameter <= 1e-8:
        raise ValueError("inserted object is too small to frame")

    data = bpy.data.cameras.new("LC_CompositionCamera")
    camera = bpy.data.objects.new("LC_CompositionCamera", data)
    bpy.context.collection.objects.link(camera)
    camera.data.lens = focal_length
    bpy.context.scene.camera = camera

    rng = random.Random(int(job["seed"]) ^ 0x4C4343414D455241)
    coarse_candidates = [(azimuth, elevation) for elevation in elevations for azimuth in azimuths]
    rng.shuffle(coarse_candidates)
    coarse_candidates = coarse_candidates[: min(candidate_count, len(coarse_candidates))]
    repair_count = int(config.get("repair_attempts_per_candidate", 6))
    if repair_count < 1:
        raise ValueError("camera repair_attempts_per_candidate must be positive")
    repairs = _camera_repair_patterns(repair_count)
    scored_candidate_limit = int(config.get("scored_candidate_limit", 6))
    if scored_candidate_limit < 1:
        raise ValueError("camera scored_candidate_limit must be positive")
    failures: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    minimum_pixels = int(render_config["minimum_visible_pixels"])
    minimum_visible_ratio = float(config.get("minimum_visible_ratio", 0.70))
    if not 0.0 < minimum_visible_ratio <= 1.0:
        raise ValueError("camera minimum_visible_ratio must be in (0, 1]")
    target_minimum_pixels = int(config.get("target_minimum_visible_pixels", 64))
    if target_minimum_pixels < 1:
        raise ValueError("camera target_minimum_visible_pixels must be positive")
    # A large support entity (for example a long table) need not fit in full;
    # for place_on, keeping the contact region visible is the useful contract.
    if job["target"]["relation"] == "place_on":
        target_points = [Vector(job["target"]["bottom_center_world"])]
    else:
        target_points = [Vector(job["target"]["center_world"])]

    scene_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj not in set(inserted_meshes) and not obj.hide_render
    ]
    bounds_cache: dict[Any, tuple[Any, Any]] = {}
    bvh_cache: dict[Any, BVHTree | None] = {}
    attempt_index = 0
    for coarse_index, (base_azimuth, base_elevation) in enumerate(coarse_candidates):
      base_fill = rng.uniform(*fill_range)
      base_shift_x = rng.uniform(*shift_x_range)
      base_shift_y = rng.uniform(*shift_y_range)
      for repair_index, repair in enumerate(repairs):
        candidate_index = attempt_index
        attempt_index += 1
        azimuth = (base_azimuth + repair["azimuth_offset"]) % 360.0
        elevation = min(80.0, max(5.0, base_elevation + repair["elevation_offset"]))
        desired_fill = base_fill
        azimuth_radians = math.radians(azimuth)
        elevation_radians = math.radians(elevation)
        direction = Vector(
            (
                math.cos(elevation_radians) * math.cos(azimuth_radians),
                math.cos(elevation_radians) * math.sin(azimuth_radians),
                math.sin(elevation_radians),
            )
        ).normalized()
        camera.data.shift_x = base_shift_x * repair["shift_scale"]
        camera.data.shift_y = base_shift_y * repair["shift_scale"]
        view_angle = min(float(camera.data.angle_x), float(camera.data.angle_y))
        distance = (subject_diameter * 0.5) / math.tan(view_angle * desired_fill * 0.5)
        distance = max(distance, subject_diameter * 1.05, 0.25)
        camera.location = anchor + direction * distance
        _look_at(camera, anchor)
        bpy.context.view_layer.update()

        # Correct the first-order distance estimate using the actual projected bounds.
        projected = _project_objects(camera, inserted_meshes)
        actual_fill = max(projected["width"], projected["height"])
        if actual_fill > 1e-6:
            distance *= actual_fill / desired_fill
            distance *= repair["distance_multiplier"]
            camera.location = anchor + direction * distance
            _look_at(camera, anchor)
            bpy.context.view_layer.update()
            projected = _project_objects(camera, inserted_meshes)

        target_ndc = [_project_point(camera, point) for point in target_points]
        reason = _camera_rejection_reason(
            projected,
            target_ndc,
            fill_range,
            ndc_x_range,
            ndc_y_range,
            edge_margin,
        )
        if reason is not None:
            failures.append({
                "candidate": candidate_index,
                "coarse_candidate": coarse_index,
                "repair": repair_index,
                "reason": reason,
            })
            continue
        clearance = _camera_clearance(
            camera.location,
            scene_meshes,
            subject_diameter,
            config,
            bounds_cache,
            bvh_cache,
        )
        required_clearance = max(
            float(config.get("minimum_clearance", 0.05)),
            subject_diameter * float(config.get("clearance_subject_ratio", 0.10)),
        )
        if clearance < required_clearance:
            failures.append({
                "candidate": candidate_index,
                "coarse_candidate": coarse_index,
                "repair": repair_index,
                "reason": f"camera_clearance={clearance:.6g} < {required_clearance:.6g}",
            })
            continue
        foreground_fraction = _near_foreground_fraction(
            camera,
            anchor,
            inserted_meshes,
            subject_diameter,
            config,
        )
        maximum_foreground = float(config.get("maximum_foreground_fraction", 0.35))
        if foreground_fraction > maximum_foreground:
            failures.append({
                "candidate": candidate_index,
                "coarse_candidate": coarse_index,
                "repair": repair_index,
                "reason": (
                    f"near_foreground_fraction={foreground_fraction:.4f} "
                    f"> {maximum_foreground:.4f}"
                ),
            })
            continue
        mask_path = partial / "diagnostics" / f"camera_candidate_{candidate_index:03d}.png"
        visible_pixels = _render_mask(mask_path, inserted_meshes, render_config)
        if visible_pixels < minimum_pixels:
            failures.append(
                {
                    "candidate": candidate_index,
                    "reason": f"visible_pixels={visible_pixels} < {minimum_pixels}",
                }
            )
            continue
        reference_mask_path = (
            partial / "diagnostics" / f"camera_candidate_{candidate_index:03d}_reference.png"
        )
        reference_pixels = _render_mask(
            reference_mask_path,
            inserted_meshes,
            render_config,
            occlude=False,
        )
        visible_ratio = visible_pixels / max(reference_pixels, 1)
        if visible_ratio < minimum_visible_ratio:
            failures.append({
                "candidate": candidate_index,
                "coarse_candidate": coarse_index,
                "repair": repair_index,
                "reason": (
                    f"visible_ratio={visible_ratio:.4f} < {minimum_visible_ratio:.4f}"
                ),
            })
            continue
        if job["target"]["relation"] == "place_on":
            visible_target_meshes = [
                obj for obj in target_objects if obj.type == "MESH" and not obj.hide_render
            ]
            if not visible_target_meshes:
                failures.append(
                    {"candidate": candidate_index, "reason": "place_on target has no visible mesh"}
                )
                continue
            target_mask_path = (
                partial / "diagnostics" / f"camera_target_{candidate_index:03d}.png"
            )
            target_visible_pixels = _render_mask(
                target_mask_path, visible_target_meshes, render_config
            )
            if target_visible_pixels < target_minimum_pixels:
                failures.append(
                    {
                        "candidate": candidate_index,
                        "reason": (
                            f"target_visible_pixels={target_visible_pixels} "
                            f"< {target_minimum_pixels}"
                        ),
                    }
                )
                continue
        else:
            # replace hides the original target; the inserted object is its visible proxy.
            target_visible_pixels = visible_pixels
        fixture_mask_path = None
        fixture_visible_pixels = None
        fixture_screen_fraction = None
        if fixture is not None:
            fixture_objects = [
                obj for obj in fixture.get("objects", []) if obj.type == "MESH" and not obj.hide_render
            ]
            if not fixture_objects:
                failures.append({
                    "candidate": candidate_index,
                    "coarse_candidate": coarse_index,
                    "repair": repair_index,
                    "reason": "fixture has no visible mesh",
                })
                continue
            fixture_mask_path = (
                partial / "diagnostics" / f"camera_fixture_{candidate_index:03d}.png"
            )
            fixture_visible_pixels = _render_mask(
                fixture_mask_path, fixture_objects, render_config
            )
            fixture_settings = fixture_config or {}
            fixture_minimum = int(fixture_settings.get("minimum_visible_pixels", 64))
            fixture_screen_fraction = fixture_visible_pixels / float(
                int(render_config["resolution"]) ** 2
            )
            fixture_maximum = float(fixture_settings.get("maximum_screen_fraction", 0.05))
            if fixture_visible_pixels < fixture_minimum or fixture_screen_fraction > fixture_maximum:
                failures.append({
                    "candidate": candidate_index,
                    "coarse_candidate": coarse_index,
                    "repair": repair_index,
                    "reason": (
                        f"fixture pixels/fraction {fixture_visible_pixels}/"
                        f"{fixture_screen_fraction:.4f} outside contract"
                    ),
                })
                continue
        fill = max(projected["width"], projected["height"])
        fill_midpoint = (fill_range[0] + fill_range[1]) * 0.5
        score = (
            visible_ratio * 4.0
            - foreground_fraction * 3.0
            - abs(fill - fill_midpoint)
            + min(clearance / max(subject_diameter, 1e-8), 2.0) * 0.1
        )
        accepted.append({
            "score": score,
            "location": camera.location.copy(),
            "rotation_euler": camera.rotation_euler.copy(),
            "shift_x": float(camera.data.shift_x),
            "shift_y": float(camera.data.shift_y),
            "result": {
            "candidate_index": candidate_index,
            "coarse_candidate_index": coarse_index,
            "repair_index": repair_index,
            "azimuth_degrees": azimuth,
            "elevation_degrees": elevation,
            "visible_pixels": visible_pixels,
            "reference_pixels": reference_pixels,
            "visible_ratio": visible_ratio,
            "camera_clearance": clearance,
            "near_foreground_fraction": foreground_fraction,
            "target_visible_pixels": target_visible_pixels,
            "mask_path": mask_path,
            "reference_mask_path": reference_mask_path,
            "subject_diameter": subject_diameter,
            "inserted_ndc_bounds": projected,
            "target_entity_ndc": target_ndc,
            "fixture_mask_path": fixture_mask_path,
            "fixture_visible_pixels": fixture_visible_pixels,
            "fixture_screen_fraction": fixture_screen_fraction,
            },
        })
        # One locally repaired result per coarse orbit is enough for the
        # bounded quality pool; keep searching other viewpoints rather than
        # rendering every nearby variant.
        break
      if len(accepted) >= scored_candidate_limit:
        break

    diagnostics_root = partial / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    failure_path = diagnostics_root / "camera_failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if accepted:
        accepted.sort(key=lambda row: (-row["score"], row["result"]["candidate_index"]))
        best = accepted[0]
        camera.location = best["location"]
        camera.rotation_euler = best["rotation_euler"]
        camera.data.shift_x = best["shift_x"]
        camera.data.shift_y = best["shift_y"]
        bpy.context.view_layer.update()
        return camera, anchor, best["result"]
    raise ValueError(
        f"no generated camera kept the composition target visible; "
        f"candidates={attempt_index}, diagnostics={failure_path}"
    )


def _camera_repair_patterns(count: int) -> list[dict[str, float]]:
    patterns = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 10.0, 1.05, 0.85),
        (-10.0, 5.0, 1.12, 0.80),
        (10.0, 5.0, 1.12, 0.80),
        (-20.0, 10.0, 1.25, 0.65),
        (20.0, 10.0, 1.25, 0.65),
        (0.0, 20.0, 1.35, 0.50),
        (-30.0, 15.0, 1.40, 0.50),
        (30.0, 15.0, 1.40, 0.50),
    ]
    return [
        {
            "azimuth_offset": item[0],
            "elevation_offset": item[1],
            "distance_multiplier": item[2],
            "shift_scale": item[3],
        }
        for item in patterns[:count]
    ]


def _point_aabb_distance(point: Vector, minimum: Vector, maximum: Vector) -> float:
    squared = 0.0
    for axis in range(3):
        delta = max(float(minimum[axis]) - point[axis], 0.0, point[axis] - float(maximum[axis]))
        squared += delta * delta
    return math.sqrt(squared)


def _camera_clearance(
    location,
    scene_meshes,
    subject_diameter,
    config,
    bounds_cache,
    bvh_cache,
) -> float:
    required = max(
        float(config.get("minimum_clearance", 0.05)),
        subject_diameter * float(config.get("clearance_subject_ratio", 0.10)),
    )
    # A finite capped value keeps metadata JSON portable when no surface lies
    # inside the clearance query radius.
    best = required * 2.0
    point = Vector(location)
    for obj in scene_meshes:
        if obj not in bounds_cache:
            bounds_cache[obj] = _world_bounds([obj])
        broad_distance = _point_aabb_distance(point, *bounds_cache[obj])
        if broad_distance > min(best, required * 2.0):
            continue
        if obj not in bvh_cache:
            bvh_cache[obj] = _world_bvh(obj)
        bvh = bvh_cache[obj]
        if bvh is None:
            continue
        nearest = bvh.find_nearest(point)
        if nearest is not None:
            best = min(best, float(nearest[3]))
    return best


def _near_foreground_fraction(camera, anchor, inserted_meshes, subject_diameter, config) -> float:
    grid = int(config.get("foreground_grid_size", 12))
    if grid < 2:
        raise ValueError("camera foreground_grid_size must be at least 2")
    maximum_distance = (Vector(anchor) - camera.location).length
    depth_margin = subject_diameter * float(config.get("foreground_depth_margin_ratio", 0.20))
    inserted_names = {obj.name for obj in inserted_meshes}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rotation = camera.matrix_world.to_quaternion()
    tan_x = math.tan(float(camera.data.angle_x) * 0.5)
    tan_y = math.tan(float(camera.data.angle_y) * 0.5)
    near_hits = 0
    total = grid * grid
    for row in range(grid):
        v = (row + 0.5) / grid
        for column in range(grid):
            u = (column + 0.5) / grid
            x = (2.0 * (u - 0.5 + float(camera.data.shift_x))) * tan_x
            y = (2.0 * (v - 0.5 + float(camera.data.shift_y))) * tan_y
            direction = rotation @ Vector((x, y, -1.0)).normalized()
            hit, location, _normal, _face, obj, _matrix = bpy.context.scene.ray_cast(
                depsgraph,
                camera.location,
                direction,
                distance=maximum_distance,
            )
            if not hit or obj is None or obj.name in inserted_names:
                continue
            if (location - camera.location).length < maximum_distance - depth_margin:
                near_hits += 1
    return near_hits / total


def _project_objects(camera, objects) -> dict[str, float]:
    points = [
        world_to_camera_view(bpy.context.scene, camera, obj.matrix_world @ Vector(corner))
        for obj in objects
        for corner in obj.bound_box
    ]
    if not points:
        raise ValueError("cannot project an empty object set")
    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    depths = [float(point.z) for point in points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "center_x": (min(xs) + max(xs)) * 0.5,
        "center_y": (min(ys) + max(ys)) * 0.5,
        "min_depth": min(depths),
    }


def _project_point(camera, point) -> list[float]:
    ndc = world_to_camera_view(bpy.context.scene, camera, point)
    return [float(ndc.x), float(ndc.y), float(ndc.z)]


def _camera_rejection_reason(projected, target_ndc, fill_range, ndc_x, ndc_y, margin):
    if projected["min_depth"] <= 0:
        return "inserted object crosses or is behind the camera plane"
    if projected["min_x"] < margin or projected["max_x"] > 1.0 - margin:
        return "inserted object is horizontally cropped"
    if projected["min_y"] < margin or projected["max_y"] > 1.0 - margin:
        return "inserted object is vertically cropped"
    fill = max(projected["width"], projected["height"])
    if not fill_range[0] <= fill <= fill_range[1]:
        return f"subject fill {fill:.4f} is outside {fill_range}"
    if not ndc_x[0] <= projected["center_x"] <= ndc_x[1]:
        return "inserted object center is outside the configured horizontal framing range"
    if not ndc_y[0] <= projected["center_y"] <= ndc_y[1]:
        return "inserted object center is outside the configured vertical framing range"
    for point in target_ndc:
        if point[2] <= 0 or not ndc_x[0] <= point[0] <= ndc_x[1] or not ndc_y[0] <= point[1] <= ndc_y[1]:
            return "target entity reference point is outside the camera framing range"
    return None


def _ordered_pair(value, label: str, lower_limit: float, upper_limit: float):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    result = [float(item) for item in value]
    if result[0] > result[1] or result[0] < lower_limit or result[1] > upper_limit:
        raise ValueError(f"{label} must be ordered inside [{lower_limit}, {upper_limit}]")
    return result


def _select_or_create_fixture(
    job, runtime, camera, camera_target, subject_diameter, partial, helpers, rng
):
    config = runtime.get("fixture", {})
    resolution = int(_render_config(runtime.get("render", {}))["resolution"])
    maximum_screen_fraction = float(config.get("maximum_screen_fraction", 0.05))
    if not 0.0 < maximum_screen_fraction <= 1.0:
        raise ValueError("fixture maximum_screen_fraction must be in (0, 1]")
    candidates: list[dict[str, Any]] = []
    for entity_id in job.get("fixture_candidate_entity_ids", []):
        objects = [obj for obj in _entity_objects(entity_id) if obj.type == "MESH" and not obj.hide_render]
        if not objects:
            continue
        mask_path = partial / "diagnostics" / f"fixture_candidate_{_safe_name(entity_id)}.png"
        pixels = _render_mask(mask_path, objects, _render_config(runtime.get("render", {})))
        if pixels < int(config.get("minimum_visible_pixels", 64)):
            continue
        screen_fraction = pixels / float(resolution**2)
        center = sum((_object_center(obj) for obj in objects), Vector()) / len(objects)
        ndc = world_to_camera_view(bpy.context.scene, camera, center)
        center_distance = math.hypot(float(ndc.x) - 0.5, float(ndc.y) - 0.5)
        candidates.append(
            {
                "source": "scene_native",
                "entity_id": entity_id,
                "objects": objects,
                "world_position": center,
                "visible_pixels": pixels,
                "screen_fraction": screen_fraction,
                "center_distance": center_distance,
                "candidate_mask": mask_path,
                "material": None,
            }
        )
    if candidates:
        # A visible real fixture always has priority over an artificial one.
        # Prefer an already well-sized native fixture; if all are oversized,
        # keep the smallest and let the final camera repair move away from it.
        candidates.sort(
            key=lambda row: (
                row["screen_fraction"] > maximum_screen_fraction,
                row["screen_fraction"]
                if row["screen_fraction"] > maximum_screen_fraction
                else -row["visible_pixels"],
                row["center_distance"],
                row["entity_id"],
            )
        )
        selected = candidates[0]
        selected["position"] = _world_to_canonical(camera, camera_target, selected["world_position"])
        return selected

    if config.get("fallback_position_mode", "subject_relative") != "subject_relative":
        raise ValueError("fixture fallback_position_mode must be subject_relative")
    ranges = config.get(
        "fallback_position_ranges",
        {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [0.8, 2.0]},
    )
    candidate_count = int(config.get("fallback_candidates", 24))
    if candidate_count < 1:
        raise ValueError("fixture fallback_candidates must be positive")
    size = max(
        float(config.get("fallback_fixture_size_min", 0.01)),
        min(
            float(config.get("fallback_fixture_size_max", 0.08)),
            subject_diameter * float(config.get("fallback_fixture_size_ratio", 0.12)),
        ),
    )
    minimum_pixels = int(config.get("minimum_visible_pixels", 64))
    for candidate_index in range(candidate_count):
        target_relative = [
            rng.uniform(*ranges[axis]) * subject_diameter for axis in ("x", "y", "z")
        ]
        # Generate the fallback independently of the provisional camera.  This
        # keeps it fixed while the final camera is moved and repaired.
        world_position = Vector(camera_target) + Vector(target_relative)
        ndc = world_to_camera_view(bpy.context.scene, camera, world_position)
        if float(ndc.z) <= 0 or not 0.05 <= float(ndc.x) <= 0.95 or not 0.05 <= float(ndc.y) <= 0.95:
            continue
        mesh, material = helpers.add_fixture(
            f"LC_ProceduralFixture_{candidate_index:03d}",
            world_position,
            size,
        )
        fixture_collisions = _find_scene_collisions(
            [mesh],
            [
                obj
                for obj in bpy.data.objects
                if obj.type == "MESH" and obj is not mesh and not obj.hide_render
            ],
            _render_config(runtime.get("render", {})),
        )
        if fixture_collisions:
            _remove_fixture_mesh(mesh, material)
            continue
        mask_path = partial / "diagnostics" / f"fixture_fallback_{candidate_index:03d}.png"
        pixels = _render_mask(mask_path, [mesh], _render_config(runtime.get("render", {})))
        screen_fraction = pixels / float(resolution**2)
        if pixels >= minimum_pixels and screen_fraction <= maximum_screen_fraction:
            return {
                "source": "procedural_fallback",
                "entity_id": None,
                "objects": [mesh],
                "world_position": world_position,
                "position": None,
                "target_relative_world": target_relative,
                "visible_pixels": pixels,
                "screen_fraction": screen_fraction,
                "candidate_mask": mask_path,
                "material": material,
            }
        _remove_fixture_mesh(mesh, material)
    raise ValueError(
        f"no procedural fallback fixture passed the visibility gate after {candidate_count} candidates"
    )


def _remove_fixture_mesh(mesh, material) -> None:
    mesh_data = mesh.data
    bpy.data.objects.remove(mesh, do_unlink=True)
    if mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)
    if material is not None and material.users == 0:
        bpy.data.materials.remove(material)


def _render_fixture_component(fixture, partial, output_root, final_directory, config, helpers):
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
        helpers.render_exr(component_path)
    finally:
        if fixture.get("material") is not None:
            helpers.set_fixture_emission(fixture["material"], (1.0, 1.0, 1.0, 1.0), 0.0)
        helpers.remove_light(light)
    return {
        "path": _future_relative(component_path, partial, final_directory, output_root),
        "mask": _future_relative(mask_path, partial, final_directory, output_root),
        "position": [float(value) for value in fixture["position"]],
        "renderer_position": [float(value) for value in fixture["world_position"]],
        "base_energy": energy,
        "diffuse": radius,
        "fixture_source": fixture["source"],
        "fixture_entity_id": fixture["entity_id"],
        "visible_pixels": int(fixture["visible_pixels"]),
        "screen_fraction": float(fixture["screen_fraction"]),
        "binding": "controlled_point_at_fixture_center",
    }


def _render_point_components(partial, output_root, final_directory, camera, target, config, helpers, rng):
    count = int(config["point_lights_per_scene"])
    if count < 1:
        return []
    light = helpers.add_light("LC_PointLight", "POINT", (0, 0, 0), float(config["point_energy"]), 0.0)
    rows = []
    try:
        for index in range(count):
            position = [rng.uniform(*config["point_position_ranges"][axis]) for axis in ("x", "y", "z")]
            world_position = helpers.camera_to_world(camera, target, position)
            radius = rng.uniform(*config["point_radius_range"])
            light.location = world_position
            light.data.shadow_soft_size = radius
            bpy.context.view_layer.update()
            path = partial / "point_lights" / f"light_{index:03d}.exr"
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


def _render_diffuse_components(partial, output_root, final_directory, camera, target, config, helpers):
    spreads = [float(value) for value in config["diffuse_sizes"]]
    if len(spreads) < 2:
        raise ValueError("at least two diffuse sizes are required")
    world_position = helpers.camera_to_world(camera, target, config["diffuse_position"])
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


def _render_mask(
    path: Path,
    selected_objects,
    config: dict[str, Any],
    *,
    occlude: bool = True,
) -> int:
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
            if not occlude and obj not in selected:
                obj.hide_render = True
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
    world = bpy.data.worlds.new("LC_ControlledWorld")
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
        print(f"COMPOSITION_HDRI {job['job_id']}: {hdri}", flush=True)
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
        f"{job['seed']}\0{job['job_id']}\0ambient-hdri-v1".encode("utf-8")
    ).digest()
    return candidates[int.from_bytes(selector[:8], "big") % len(candidates)]


@lru_cache(maxsize=8)
def _ambient_hdri_candidates(root_value: str) -> tuple[Path, ...]:
    root = Path(root_value)
    return tuple(sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".hdr", ".exr"}
    ))


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
        "minimum_visible_pixels": 256,
        "place_footprint_limit": 1.0,
        "collision_aabb_tolerance": 0.002,
        "collision_bvh_epsilon": 0.0,
        "collision_shrink_enabled": True,
        "collision_shrink_factor": 0.9,
        "collision_min_scale_ratio": 0.6,
        "collision_max_attempts": 6,
        "point_lights_per_scene": 16,
        "point_energy": 500.0,
        "point_position_ranges": {
            "x": [-1.2, 1.2],
            "y": [-0.6, 1.2],
            "z": [0.15, 1.8],
        },
        "point_radius_range": [0.03, 0.25],
        "diffuse_sizes": [0.05, 0.2, 0.4, 0.7, 1.0, 1.5],
        "diffuse_position": [0.0, 0.3, 1.3],
        "fixture_energy": 300.0,
        "fixture_light_radius": 0.12,
        "fallback_emission_strength": 4.0,
        "require_verified_license": True,
    }
    result = {**defaults, **(value or {})}
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
                "light_energy": float(obj.data.energy) if obj.type == "LIGHT" else None,
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


def _object_center(obj) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def _camera_intrinsics(camera) -> dict[str, float | int]:
    """Pinhole intrinsics for the OpenCV camera frame.

    ``u = fx * X / Z + cx`` and ``v = fy * Y / Z + cy`` with ``X`` right, ``Y``
    down, ``Z`` the forward depth and the pixel origin at the top-left corner.
    Blender's ``shift_x``/``shift_y`` move the frame instead of rotating the
    camera, so they offset the principal point rather than the focal length.
    """
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


def _world_to_canonical(camera, target, position) -> list[float]:
    forward = (Vector(target) - camera.location).normalized()
    right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
    up = right.cross(forward).normalized()
    delta = Vector(position) - Vector(target)
    return [float(delta.dot(right)), float(delta.dot(forward)), float(delta.dot(up))]


def _axis_vector(axis: str) -> Vector:
    sign = -1.0 if axis.startswith("-") else 1.0
    base = {
        "X": Vector((1.0, 0.0, 0.0)),
        "Y": Vector((0.0, 1.0, 0.0)),
        "Z": Vector((0.0, 0.0, 1.0)),
    }[axis[-1]]
    return base * sign


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


if __name__ == "__main__":
    main()
