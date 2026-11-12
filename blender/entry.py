from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
# Blender runs this file by path with its own Python, so register the package
# under its canonical name: the checkout folder may have any name.
from _launcher import ensure_package  # noqa: E402

ensure_package()

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from datacompose.scene_semantics import (  # noqa: E402
    category_has_affordance,
    classify_scene_name,
    normalize_label,
)


def main() -> None:
    arguments = _parse_args()
    if arguments.command == "extract":
        _extract(Path(arguments.job))
    elif arguments.command == "review":
        _render_review(Path(arguments.job))
    else:  # pragma: no cover - argparse enforces command values.
        raise ValueError(arguments.command)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Blender-side LightConstruction worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--job", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--job", required=True)
    return parser.parse_args(argv)


def _extract(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    _clear_startup_scene()
    if job.get("ignore_source_lights"):
        _install_ignore_light_reader()

    result = bpy.ops.import_scene.fbx(
        filepath=job["source_fbx"],
        use_custom_props=True,
        use_image_search=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"FBX import did not finish: {sorted(result)}")

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    bpy.context.view_layer.update()

    aliases = job.get("aliases", {})
    rename_overrides = job.get("rename_overrides", {})
    support_categories = set(job.get("support_categories", []))
    non_replaceable_categories = set(job.get("non_replaceable_categories", []))
    hash_length = int(job.get("id_hash_length", 16))
    nodes: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    imported_objects = sorted(scene.objects, key=lambda item: (item.type, item.name))
    for obj in imported_objects:
        object_id = _stable_id(
            job["scene_id"], "node", job["source_relative"], obj.type, obj.name, length=hash_length
        )
        obj["lc_scene_object_id"] = object_id
        obj["lc_raw_import_name"] = obj.name
        obj["lc_source_fbx"] = job["source_relative"]
        if obj.type != "MESH":
            continue

        category, raw_label, category_confidence = classify_scene_name(
            obj.name, aliases, rename_overrides
        )
        bounds = _world_bounds(obj)
        materials = sorted(
            {
                slot.material.name
                for slot in obj.material_slots
                if slot.material is not None
            }
        )
        obj.data.calc_loop_triangles()
        node = {
            "node_id": object_id,
            "raw_import_name": obj.name,
            "source_fbx_element_id": None,
            "object_type": obj.type,
            "transform_world": _matrix_flat(obj.matrix_world),
            "aabb_world": bounds["aabb"],
            "obb_world": bounds["obb"],
            "materials": materials,
            "triangle_count": len(obj.data.loop_triangles),
            "vertex_count": len(obj.data.vertices),
        }
        nodes.append(node)

        entity_id = _stable_id(
            job["scene_id"], "entity", object_id, category, length=hash_length
        )
        support_surface = category_has_affordance(category, support_categories)
        non_replaceable = category_has_affordance(category, non_replaceable_categories)
        entity = {
            "entity_id": entity_id,
            "node_ids": [object_id],
            "raw_label": raw_label,
            "category": normalize_label(category),
            "category_confidence": category_confidence,
            "grouping_confidence": 0.85,
            "grouping_method": "single_mesh_conservative",
            "replaceable": not non_replaceable,
            "support_surface": support_surface,
            "obb_world": bounds["obb"],
        }
        obj["lc_scene_entity_id"] = entity_id
        obj["lc_scene_category"] = entity["category"]
        entities.append(entity)

    normalized_blend = Path(job["normalized_blend"])
    normalized_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(normalized_blend))

    fragment = {
        "scene_id": job["scene_id"],
        "source_fbx": job["source_relative"],
        "source_digest": job["source_digest"],
        "normalized_blend": job["normalized_blend_relative"],
        "units": "meter",
        "up_axis": "+Z",
        "config_digest": job["config_digest"],
        "nodes": sorted(nodes, key=lambda item: item["node_id"]),
        "entities": sorted(entities, key=lambda item: item["entity_id"]),
        "stats": {
            "objects": len(imported_objects),
            "mesh_nodes": len(nodes),
            "entities": len(entities),
            "cameras": sum(item.type == "CAMERA" for item in imported_objects),
            "lights": sum(item.type == "LIGHT" for item in imported_objects),
        },
    }
    _atomic_json(Path(job["fragment_path"]), fragment)
    print(f"LIGHTCONSTRUCTION_SCENE_RESULT={job['fragment_path']}")


def _clear_startup_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _install_ignore_light_reader() -> None:
    # Compatibility-only mode for geometry extraction. Blender 4.5 should not need this.
    import importlib

    module = importlib.import_module("io_scene_fbx.import_fbx")

    def ignored_light(_template: Any, _element: Any, _settings: Any) -> Any:
        light = bpy.data.lights.new(name="LC_Ignored_FBX_Light", type="POINT")
        light.energy = 0.0
        return light

    module.blen_read_light = ignored_light


def _render_review(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 384
    scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True

    camera_data = bpy.data.cameras.new("LC_REVIEW_CAMERA")
    camera = bpy.data.objects.new("LC_REVIEW_CAMERA", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 55.0

    mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    object_index = {
        obj.get("lc_scene_object_id"): obj
        for obj in mesh_objects
        if obj.get("lc_scene_object_id")
    }
    for obj in mesh_objects:
        obj.color = (0.35, 0.35, 0.35, 1.0)

    output_root = Path(job["output_root"])
    for item in job["items"]:
        selected = [object_index[node_id] for node_id in item["node_ids"] if node_id in object_index]
        if not selected:
            continue
        minimum, maximum = _combined_world_aabb(selected)
        center = Vector([(minimum[index] + maximum[index]) / 2.0 for index in range(3)])
        dimensions = Vector([maximum[index] - minimum[index] for index in range(3)])
        radius = max(0.05, dimensions.length / 2.0)
        for obj in selected:
            obj.color = (0.95, 0.12, 0.45, 1.0)

        entity_root = output_root / job["scene_id"] / item["entity_id"]
        entity_root.mkdir(parents=True, exist_ok=True)
        _render_review_view(scene, camera, center, radius, "context", entity_root, mesh_objects, selected)
        _render_review_view(scene, camera, center, radius, "isolated", entity_root, mesh_objects, selected)
        _render_review_view(scene, camera, center, radius, "top", entity_root, mesh_objects, selected)

        for obj in selected:
            obj.color = (0.35, 0.35, 0.35, 1.0)
    print(f"LIGHTCONSTRUCTION_REVIEW_RESULT={output_root}")


def _render_review_view(
    scene: Any,
    camera: Any,
    center: Vector,
    radius: float,
    view: str,
    entity_root: Path,
    all_meshes: list[Any],
    selected: list[Any],
) -> None:
    isolated = view in {"isolated", "top"}
    selected_set = set(selected)
    old_visibility = {obj: obj.hide_render for obj in all_meshes}
    try:
        if isolated:
            for obj in all_meshes:
                obj.hide_render = obj not in selected_set
        if view == "top":
            camera.location = center + Vector((0.0, 0.0, max(radius * 3.0, 1.0)))
            camera.data.type = "ORTHO"
            camera.data.ortho_scale = max(radius * 2.8, 0.25)
        else:
            camera.location = center + Vector((radius * 2.2, -radius * 2.2, radius * 1.5))
            camera.data.type = "PERSP"
        _look_at(camera, center)
        scene.render.filepath = str(entity_root / f"{view}.png")
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, value in old_visibility.items():
            obj.hide_render = value


def _combined_world_aabb(objects: list[Any]) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = [min(corner[index] for corner in corners) for index in range(3)]
    maximum = [max(corner[index] for corner in corners) for index in range(3)]
    return minimum, maximum


def _look_at(camera: Any, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _world_bounds(obj: Any) -> dict[str, Any]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = [min(corner[index] for corner in corners) for index in range(3)]
    maximum = [max(corner[index] for corner in corners) for index in range(3)]
    center = [(minimum[index] + maximum[index]) / 2.0 for index in range(3)]
    dimensions = [maximum[index] - minimum[index] for index in range(3)]
    return {
        "aabb": {"min": minimum, "max": maximum},
        "obb": {
            "corners": [[float(value) for value in corner] for corner in corners],
            "center": center,
            "dimensions": dimensions,
            "transform_world": _matrix_flat(obj.matrix_world),
        },
    }


def _matrix_flat(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _stable_id(scene_id: str, kind: str, *parts: str, length: int) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:length]
    return f"{scene_id}:{kind}:{digest}"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
