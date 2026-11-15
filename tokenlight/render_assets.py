"""Blender-side render helpers for the scene-only pipeline.

Loaded dynamically by ``blender/render_scene.py``.  Every function here runs
inside Blender (``bpy`` is imported lazily); the module itself is safe to import
from regular Python for inspection.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path


def camera_to_world(camera, target, position):
    forward = (target - camera.location).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward).normalized()
    return target + right * position[0] + forward * position[1] + up * position[2]


def configure_render(config: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if config["engine"] == "EEVEE" else "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = int(config["resolution"])
    scene.render.resolution_percentage = 100
    threads_per_worker = config.get("threads_per_worker", "auto")
    if str(threads_per_worker).lower() == "auto":
        scene.render.threads_mode = "AUTO"
    else:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = int(threads_per_worker)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    if hasattr(scene.render.image_settings, "exr_codec"):
        scene.render.image_settings.exr_codec = "ZIP"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.use_persistent_data = bool(config.get("persistent_data", True))
    if hasattr(scene.render, "film_transparent"):
        scene.render.film_transparent = False
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(config["samples"])
        scene.cycles.use_denoising = bool(config["denoise"])
        if str(config["device"]).upper() == "GPU":
            try:
                preferences = bpy.context.preferences.addons["cycles"].preferences
                compute_device_type = str(config["compute_device_type"]).upper()
                preferences.compute_device_type = compute_device_type
                preferences.get_devices()
                enabled_devices = []
                for device in preferences.devices:
                    device.use = device.type == compute_device_type
                    if device.use:
                        enabled_devices.append(device.name)
                if not enabled_devices:
                    raise RuntimeError(f"没有找到 Cycles {compute_device_type} 设备")
                scene.cycles.device = "GPU"
                print(f"Cycles GPU: {', '.join(enabled_devices)}", flush=True)
            except Exception as error:
                if bool(config.get("require_gpu", False)):
                    raise RuntimeError(f"Cycles GPU 初始化失败且 require_gpu=true: {error}") from error
                print(f"Cycles GPU 初始化失败，回退 CPU: {error}", flush=True)
                scene.cycles.device = "CPU"


PASS_SOCKETS = {
    # Blender 4.5 renamed the Render Layers passes to short identifiers:
    # "Diffuse Color" -> "DiffCol", "Diffuse Direct" -> "DiffDir",
    # "Diffuse Indirect" -> "DiffInd".  Each key lists its known spellings and
    # the first available socket wins, so 4.2-4.5 all resolve.
    "albedo": ("Diffuse Color", "DiffCol"),
    "normal_world": ("Normal",),
    "direct": ("Diffuse Direct", "DiffDir"),
    "indirect": ("Diffuse Indirect", "DiffInd"),
    "color": ("Diffuse Color", "DiffCol"),
}


def _pass_socket(layers, candidates):
    """Resolve a render-layer pass socket across Blender naming schemes."""
    if isinstance(candidates, str):
        candidates = (candidates,)
    for name in candidates:
        socket = layers.outputs.get(name)
        if socket is not None:
            return socket
    available = ", ".join(output.name for output in layers.outputs)
    raise RuntimeError(
        f"none of the render layer passes {tuple(candidates)} is available on "
        f"Blender {bpy.app.version_string}; available: {available}"
    )


def configure_passes(config: dict) -> None:
    """Enable Cycles passes and build the compositor outputs that write them.

    Only ``CompositorNodeOutputFile`` is used, so the graph stays compatible
    across Blender versions: every derived quantity (irradiance, camera-space
    normals) is computed in Python from the raw passes instead of with
    compositor maths.
    """
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_normal = True
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name.startswith("LC_PASS_"):
            tree.nodes.remove(node)
    layers = next((node for node in tree.nodes if node.type == "R_LAYERS"), None)
    if layers is None:
        layers = tree.nodes.new("CompositorNodeRLayers")
    for key, socket in PASS_SOCKETS.items():
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.name = f"LC_PASS_{key}"
        output.base_path = str(Path(bpy.context.blend_data.filepath or ".").parent)
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "16"
        output.format.color_mode = "RGB"
        if hasattr(output.format, "exr_codec"):
            output.format.exr_codec = "ZIP"
        output.file_slots[0].path = f"{key}_"
        output.mute = True
        tree.links.new(_pass_socket(layers, socket), output.inputs[0])


def _pass_outputs():
    scene = bpy.context.scene
    if not getattr(scene, "use_nodes", False) or scene.node_tree is None:
        return []
    return [
        node
        for node in scene.node_tree.nodes
        if node.name.startswith("LC_PASS_") and node.type == "OUTPUT_FILE"
    ]


def _mute_all_passes() -> None:
    for node in _pass_outputs():
        node.mute = True


def render_with_passes(
    passes_dir: Path,
    keys,
    *,
    combined_path: Path | None = None,
    samples: int | None = None,
    film_transparent: bool | None = None,
) -> dict[str, Path]:
    """Render one still and return the requested raw Cycles passes.

    ``combined_path`` writes the normal combined image as well; when it is
    ``None`` only the passes reach disk.  ``keys`` must be a subset of
    :data:`PASS_SOCKETS`.
    """
    scene = bpy.context.scene
    outputs = {node.name[len("LC_PASS_") :]: node for node in _pass_outputs()}
    missing = [key for key in keys if key not in outputs]
    if missing:
        raise RuntimeError(
            f"pass outputs are not configured (missing {missing}); call configure_passes first"
        )
    passes_dir.mkdir(parents=True, exist_ok=True)
    if combined_path is not None:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "filepath": scene.render.filepath,
        "film_transparent": scene.render.film_transparent,
        "samples": scene.cycles.samples if scene.render.engine == "CYCLES" else None,
        "denoising": scene.cycles.use_denoising if scene.render.engine == "CYCLES" else None,
    }
    try:
        if samples is not None and scene.render.engine == "CYCLES":
            scene.cycles.samples = int(samples)
        if film_transparent is not None:
            scene.render.film_transparent = bool(film_transparent)
        active = set(keys)
        for key, node in outputs.items():
            node.mute = key not in active
            node.base_path = str(passes_dir)
            node.file_slots[0].path = f"{key}_"
        scene.render.filepath = str(
            combined_path if combined_path is not None else passes_dir / "_unused.exr"
        )
        bpy.ops.render.render(write_still=combined_path is not None)
        result: dict[str, Path] = {}
        for key in keys:
            matches = sorted(passes_dir.glob(f"{key}_*.exr"))
            if not matches:
                raise RuntimeError(f"pass {key!r} was not written to {passes_dir}")
            target = passes_dir / f"{key}.exr"
            if target.exists():
                target.unlink()
            matches[-1].rename(target)
            for leftover in matches[:-1]:
                leftover.unlink(missing_ok=True)
            result[key] = target
        return result
    finally:
        scene.render.filepath = saved["filepath"]
        scene.render.film_transparent = saved["film_transparent"]
        if scene.render.engine == "CYCLES":
            if saved["samples"] is not None:
                scene.cycles.samples = saved["samples"]
            if saved["denoising"] is not None:
                scene.cycles.use_denoising = saved["denoising"]
        _mute_all_passes()


_INTRINSIC_MODULE = None


def _load_intrinsic_module():
    """Load ``datacompose/intrinsic.py`` without importing the package.

    The Blender-side renderer is executed by path, so ``datacompose`` is not on
    ``sys.path``.  Loading the tested pure-NumPy helpers by file keeps a single
    implementation of the irradiance maths.
    """
    global _INTRINSIC_MODULE
    if _INTRINSIC_MODULE is None:
        import importlib.util as _importlib_util

        module_path = Path(__file__).resolve().parents[1] / "intrinsic.py"
        spec = _importlib_util.spec_from_file_location(
            "datacompose_intrinsic_render_assets", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load intrinsic helpers: {module_path}")
        module = _importlib_util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _INTRINSIC_MODULE = module
    return _INTRINSIC_MODULE


def cleanup_pass_temporaries(passes_dir: Path) -> bool:
    """Remove temporary pass output while final products stay in place."""
    raw_dir = Path(passes_dir) / "_raw"
    if not raw_dir.exists():
        return True
    if raw_dir.is_dir():
        for path in sorted(raw_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
        raw_dir.rmdir()
    return not raw_dir.exists()


def move_pass_outputs(
    passes_dir: Path,
    output_root: Path,
    material_audit: dict,
    *,
    albedo: bool,
    normal: bool,
    save_normal_world: bool,
    rendered: tuple[str, ...] = ("albedo", "normal_world"),
) -> dict[str, str]:
    """Move only enabled reference products out of ``_raw`` and clean it."""
    passes_dir = Path(passes_dir)
    output_root = Path(output_root)
    raw_dir = passes_dir / "_raw" / "reference"
    result: dict[str, str] = {}
    catalog: list[tuple[str, str, bool]] = []
    if "albedo" in rendered:
        catalog.append(("albedo", "albedo.exr", albedo))
    if "normal_world" in rendered:
        catalog.append(("normal_world", "normal_world.exr", normal and save_normal_world))
    audit_path = passes_dir / "material_audit.json"
    audit_path.write_text(
        json.dumps(material_audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result["material_audit"] = audit_path.relative_to(output_root).as_posix()
    if save_normal_world and not normal:
        raise ValueError("passes.save_normal_world=true requires passes.normal=true")
    try:
        for key, filename, publish in catalog:
            source = raw_dir / (key + ".exr")
            if publish:
                if not source.is_file():
                    raise RuntimeError(f"reference pass {key!r} is missing: {source}")
                target = passes_dir / filename
                target.unlink(missing_ok=True)
                source.rename(target)
                result[key] = target.relative_to(output_root).as_posix()
            else:
                source.unlink(missing_ok=True)
        return result
    finally:
        if not cleanup_pass_temporaries(passes_dir):
            raise RuntimeError(f"pass temporary files remain: {passes_dir / '_raw'}")


def render_pass_reference(
    directory: Path,
    config: dict,
    *,
    save: tuple[str, ...] = ("albedo", "normal_world"),
) -> dict[str, Path]:
    """Render albedo and world normals once into the pass temporary tree."""
    raw_dir = Path(directory) / "_raw" / "reference"
    return render_with_passes(
        raw_dir,
        [key for key in save if key in PASS_SOCKETS],
        samples=int(config.get("pass_samples", 1)),
        film_transparent=True,
    )

def read_linear_image(path: Path):
    """Load an EXR as a top-down float32 HxWx3 array (Blender stores rows bottom-up)."""
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError(
                f"cannot read image size from {path} ({Path(path).stat().st_size} bytes)"
            )
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        pixels = buffer.reshape(height, width, 4)
        return np.flipud(pixels)[:, :, :3].copy()
    finally:
        bpy.data.images.remove(image)


@contextmanager
def _render_image_settings(file_format: str, color_mode: str, color_depth: str):
    """Temporarily point the scene's render settings at the format we want.

    ``Image.save_render`` honours ``scene.render.image_settings`` (not the file
    extension), so the settings must match the file being written.  Otherwise a
    ``valid_mask.png`` would silently contain 16-bit EXR bytes.
    """
    settings = bpy.context.scene.render.image_settings
    saved = (settings.file_format, settings.color_mode, settings.color_depth)
    try:
        settings.file_format = file_format
        settings.color_mode = color_mode
        settings.color_depth = color_depth
        yield
    finally:
        settings.file_format = saved[0]
        settings.color_mode = saved[1]
        settings.color_depth = saved[2]


def save_linear_image(path: Path, array) -> None:
    """Write a top-down float32 HxWx3 array as a 16-bit linear EXR."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {data.shape}")
    height, width = data.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = data
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=False, float_buffer=True)
    try:
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(np.flipud(rgba).ravel())
        with _render_image_settings("OPEN_EXR", "RGB", "16"):
            image.save_render(str(path), scene=bpy.context.scene)
    finally:
        bpy.data.images.remove(image)


def save_mask_image(path: Path, mask) -> None:
    """Write a boolean/sampled mask as an 8-bit grayscale PNG."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(mask).astype(np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected HxW mask, got {values.shape}")
    height, width = values.shape
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, :3] = values[:, :, None]
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(np.flipud(rgba).ravel())
        with _render_image_settings("PNG", "BW", "8"):
            image.save_render(str(path), scene=bpy.context.scene)
    finally:
        bpy.data.images.remove(image)


def _reachable_image_node(principled):
    """Return the first Image Texture node feeding ``principled`` Base Color.

    Walks upstream through node links so a Base Color driven by a small node
    group still resolves to its image.  Returns ``None`` when the chain does not
    end at an image (e.g. a pure colour or an opaque node group).
    """
    base = principled.inputs.get("Base Color") if principled is not None else None
    if base is None or not base.is_linked:
        return None
    seen = set()
    stack = [link.from_node for link in base.links]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.type == "TEX_IMAGE":
            return node
        for socket in getattr(node, "inputs", []):
            for link in socket.links:
                stack.append(link.from_node)
    return None


def _image_state(image) -> tuple[bool, list[int]]:
    """Return ``(exists, [width, height])`` for a Blender image datablock."""
    path = bpy.path.abspath(image.filepath) if image.filepath else ""
    exists = bool(image.packed_file) or (bool(path) and Path(path).is_file())
    return exists, [int(image.size[0]), int(image.size[1])]


def audit_materials() -> dict:
    """Summarise whether materials actually link their textures.

    A material counts as *textured* when its node tree contains an Image Texture
    node.  ``base_color_linked_ratio`` is computed over that subset only, so
    deliberately flat materials never look like failures.  ``base_color_linked``
    additionally requires the Base Color image to exist on disk with a non-zero
    size, treating a missing/empty placeholder as "not connected".
    """
    rows: list[dict] = []
    textured = 0
    base_linked = 0
    linked_without_image = 0
    unknown = 0
    missing: list[dict] = []
    for material in bpy.data.materials:
        entry: dict = {"material": material.name, "has_nodes": bool(material.use_nodes)}
        if not material.use_nodes or material.node_tree is None:
            rows.append(entry)
            continue
        nodes = material.node_tree.nodes
        image_nodes = [node for node in nodes if node.type == "TEX_IMAGE"]
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        for node in image_nodes:
            image = node.image
            if image is None:
                continue
            exists, _size = _image_state(image)
            if not exists:
                path = bpy.path.abspath(image.filepath) if image.filepath else ""
                missing.append({"material": material.name, "path": path})
        base_node = _reachable_image_node(principled)
        base_image = base_node.image if base_node is not None else None
        image_exists = None
        image_size = None
        linked = False
        if base_image is not None:
            image_exists, image_size = _image_state(base_image)
            linked = bool(image_exists and image_size[0] > 0 and image_size[1] > 0)
            if not linked:
                path = bpy.path.abspath(base_image.filepath) if base_image.filepath else ""
                missing.append({"material": material.name, "path": path})
        elif principled is not None and principled.inputs.get("Base Color") is not None and principled.inputs["Base Color"].is_linked:
            # Base Color is driven by something we cannot resolve to an image.
            unknown += 1
        if image_nodes:
            textured += 1
            if linked:
                base_linked += 1
        elif linked:
            linked_without_image += 1

        def input_linked(name: str) -> bool:
            socket = principled.inputs.get(name) if principled is not None else None
            return bool(socket is not None and socket.is_linked)

        entry.update(
            {
                "principled": principled is not None,
                "base_color_linked": linked,
                "base_color_image": base_image.name if base_image is not None else None,
                "image_exists": image_exists,
                "image_size": image_size,
                "textured": bool(image_nodes),
                "normal_linked": input_linked("Normal"),
                "metallic_linked": input_linked("Metallic"),
                "roughness_linked": input_linked("Roughness"),
            }
        )
        rows.append(entry)
    denominator = max(textured - unknown, 0)
    return {
        "materials": len(bpy.data.materials),
        "textured": textured,
        "textured_materials": textured,
        "base_color_linked": base_linked,
        "base_color_linked_ratio": (base_linked / denominator) if denominator else None,
        "unknown_materials": unknown,
        "linked_without_image": linked_without_image,
        "missing_image_files": len(missing),
        "missing_examples": [item["path"] for item in missing[:10]],
        "rows": rows,
    }


def add_light(name: str, kind: str, location, energy: float, radius: float):
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    if kind == "POINT":
        data.shadow_soft_size = radius
    else:
        data.shape = "DISK"
        data.size = radius
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    return light


def add_fixture(name: str, location, size: float):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=size, location=location)
    fixture = bpy.context.object
    fixture.name = name
    material = bpy.data.materials.new(f"{name}Material")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
    principled.inputs["Roughness"].default_value = 0.35
    set_fixture_emission(material, (1.0, 1.0, 1.0, 1.0), 0.0)
    fixture.data.materials.append(material)
    return fixture, material


def set_fixture_emission(material, color, strength: float) -> None:
    principled = material.node_tree.nodes.get("Principled BSDF")
    color_input = principled.inputs.get("Emission Color")
    if color_input is None:
        color_input = principled.inputs.get("Emission")
    strength_input = principled.inputs.get("Emission Strength")
    if color_input is None:
        raise RuntimeError("当前 Blender Principled BSDF 没有 emission 输入")
    color_input.default_value = color if strength > 0 else (0.0, 0.0, 0.0, 1.0)
    if strength_input is not None:
        strength_input.default_value = strength


def remove_light(light) -> None:
    data = light.data
    bpy.data.objects.remove(light, do_unlink=True)
    if data.users == 0:
        bpy.data.lights.remove(data)


def render_exr(
    path: Path,
    pass_dir: Path | None = None,
    save: tuple[str, ...] = (),
    *,
    name: str | None = None,
    eps: float = 1e-3,
) -> dict[str, Path]:
    """Render the combined image to ``path`` and optionally persist derived passes.

    When ``pass_dir`` is set and ``save`` is non-empty, the *same* Cycles render
    also writes the raw diffuse passes, so ``irradiance`` is derived without a
    second render.  Raw passes are deleted immediately; only the derived
    products listed in ``save`` survive (``irradiance`` ->
    ``<pass_dir>/irradiance/<name>.exr``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if pass_dir is None or not save:
        _mute_all_passes()
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return {}
    pass_dir = Path(pass_dir)
    label = name or path.stem
    raw_dir = pass_dir / "_raw" / label
    cleanup_pass_temporaries(pass_dir)
    raw = render_with_passes(raw_dir, ["color", "direct", "indirect"], combined_path=path)
    saved: dict[str, Path] = {}
    try:
        if "irradiance" in save:
            intrinsic = _load_intrinsic_module()
            albedo = read_linear_image(raw["color"])
            direct = read_linear_image(raw["direct"])
            indirect = read_linear_image(raw["indirect"])
            irradiance, _valid = intrinsic.derive_irradiance(
                albedo, direct, indirect, eps=float(eps)
            )
            target = pass_dir / "irradiance" / f"{label}.exr"
            save_linear_image(target, irradiance)
            saved["irradiance"] = target
    finally:
        for leftover in raw_dir.glob("*.exr"):
            leftover.unlink()
        if raw_dir.is_dir():
            for child in raw_dir.iterdir():
                child.unlink()
            raw_dir.rmdir()
        raw_parent = raw_dir.parent
        if raw_parent.is_dir() and not any(raw_parent.iterdir()):
            raw_parent.rmdir()
    return saved


try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


