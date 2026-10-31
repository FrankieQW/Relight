"""Render linear TokenLight components from standalone 3D assets.

Run with the regular project Python. This process reads config.yaml and starts
Blender with an internal JSON snapshot, so Blender does not need PyYAML.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue
import random
import subprocess
import sys
from threading import Thread


SUPPORTED_ASSETS = {".glb", ".gltf", ".fbx", ".obj", ".blend"}
SUPPORTED_HDRIS = {".hdr", ".exr"}
PROGRESS_PREFIX = "__TOKENLIGHT_PROGRESS__"


def parse_outer_args():
    parser = argparse.ArgumentParser(description="Render TokenLight components using the single config.yaml")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def launch_blender(config_path: str) -> None:
    import yaml

    source = Path(config_path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)
    runtime = dict(full_config["render"])
    runtime.update(
        {
            "asset_root": full_config["paths"]["object_root"],
            "hdri_root": full_config["paths"].get("hdri_root"),
            "output_root": full_config["paths"]["render_output_root"],
            "source_config": str(source),
        }
    )
    output_root = Path(runtime["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot = output_root / "render_config.snapshot.json"
    snapshot.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    asset_root = Path(runtime["asset_root"]).expanduser().resolve()
    assets = scan_files(asset_root, SUPPORTED_ASSETS)
    limit = runtime.get("asset_limit")
    if limit is not None:
        assets = assets[: int(limit)]
    if not assets:
        raise RuntimeError(f"没有找到支持的 3D asset: {asset_root}")
    gpu_ids = runtime.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("render.gpu_ids 必须是非空 GPU 编号列表")
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    if len(set(gpu_ids)) != len(gpu_ids) or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("render.gpu_ids 必须包含互不重复的非负 GPU 编号")
    workers_per_gpu = int(runtime.get("workers_per_gpu", 1))
    if workers_per_gpu < 1:
        raise ValueError("render.workers_per_gpu 必须大于等于 1")
    worker_count = min(len(gpu_ids) * workers_per_gpu, len(assets))
    threads_per_worker = runtime.get("threads_per_worker", "auto")
    if str(threads_per_worker).lower() != "auto" and int(threads_per_worker) < 1:
        raise ValueError("render.threads_per_worker 必须为 auto 或大于等于 1 的整数")

    blender = str(runtime["blender_executable"])
    processes = []
    recent_output = {index: deque(maxlen=80) for index in range(worker_count)}
    events: Queue = Queue()
    for worker_index in range(worker_count):
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        worker_runtime = {
            **runtime,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "assigned_gpu_id": gpu_id,
        }
        worker_snapshot = output_root / f"render_config.worker_{worker_index:03d}.snapshot.json"
        worker_snapshot.write_text(json.dumps(worker_runtime, indent=2) + "\n", encoding="utf-8")
        command = [
            blender, "--background", "--python", str(Path(__file__).resolve()), "--",
            "--runtime-config", str(worker_snapshot),
        ]
        print(
            f"启动 Blender worker {worker_index + 1}/{worker_count} (GPU {gpu_id}):",
            " ".join(command),
            flush=True,
        )
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(
            command,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        processes.append((command, process))
        Thread(target=forward_output, args=(worker_index, process, events), daemon=True).start()

    failures = 0
    return_codes = {}
    try:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(assets) * int(runtime["views_per_asset"]),
            desc="render",
            unit="scene",
            dynamic_ncols=True,
        )
        while len(return_codes) < worker_count:
            kind, worker_index, value = events.get()
            if kind == "done":
                return_codes[worker_index] = int(value)
                continue
            line = value
            recent_output[worker_index].append(line.rstrip())
            if not line.startswith(PROGRESS_PREFIX):
                if line.startswith("Cycles GPU:") or "Cycles GPU 初始化失败" in line or line.startswith("Done:"):
                    tqdm.write(f"[worker {worker_index}] {line.rstrip()}")
                continue
            event = json.loads(line[len(PROGRESS_PREFIX) :])
            if event["event"] == "stage":
                progress.set_postfix(worker=worker_index, scene=event["scene"], stage=event["stage"])
            elif event["event"] == "advance":
                if event.get("failed"):
                    failures += 1
                    tqdm.write(f"FAILED worker {worker_index} {event['scene']}: {event['error']}")
                progress.update(1)
                progress.set_postfix(
                    worker=worker_index,
                    scene=event["scene"],
                    split=event["split"],
                    failed=failures,
                )
    except KeyboardInterrupt:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
        raise
    finally:
        if "progress" in locals():
            progress.close()

    failed_workers = [index for index, code in return_codes.items() if code]
    if failed_workers:
        for worker_index in failed_workers:
            print(f"Blender worker {worker_index} 末尾输出：", flush=True)
            for line in recent_output[worker_index]:
                print(line, flush=True)
        commands = [processes[index][0] for index in failed_workers]
        raise RuntimeError(f"Blender worker 失败: {failed_workers}, commands={commands}")
    merge_worker_outputs(output_root, worker_count)


def forward_output(worker_index: int, process: subprocess.Popen, events: Queue) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        events.put(("line", worker_index, line))
    events.put(("done", worker_index, process.wait()))


def merge_worker_outputs(output_root: Path, worker_count: int) -> None:
    rows = {"train": [], "validation": [], "test": []}
    errors = []
    for worker_index in range(worker_count):
        worker_root = output_root / "render_workers" / f"worker_{worker_index:03d}"
        for split in rows:
            rows[split].extend(read_jsonl(worker_root / f"{split}.jsonl"))
        errors.extend(read_jsonl(worker_root / "render_errors.jsonl"))
    for split, split_rows in rows.items():
        write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            sorted(split_rows, key=lambda row: row["id"]),
        )
    write_jsonl(output_root / "render_errors.jsonl", errors)
    print(
        f"Done: {len(rows['train'])} train, {len(rows['validation'])} validation, "
        f"{len(rows['test'])} test, {len(errors)} failed",
        flush=True,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def blender_main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    asset_root = Path(config["asset_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    configure_render(config)
    assets = scan_files(asset_root, SUPPORTED_ASSETS)
    limit = config.get("asset_limit")
    if limit is not None:
        assets = assets[: int(limit)]
    if not assets:
        raise RuntimeError(f"没有找到支持的 3D asset: {asset_root}")
    worker_index = int(config.get("worker_index", 0))
    worker_count = int(config.get("worker_count", 1))
    indexed_assets = [
        (asset_index, asset_path)
        for asset_index, asset_path in enumerate(assets)
        if asset_index % worker_count == worker_index
    ]
    hdri_root = config.get("hdri_root")
    hdri_paths = scan_files(Path(hdri_root).expanduser(), SUPPORTED_HDRIS) if hdri_root else []
    split_by_asset = assign_splits(assets, config)
    rows = {"train": [], "validation": [], "test": []}
    errors = []
    seed = int(config["split_seed"])
    for asset_index, asset_path in indexed_assets:
        split = split_by_asset[asset_path]
        for view_index in range(int(config["views_per_asset"])):
            rng = random.Random(seed + asset_index * 1000 + view_index)
            try:
                metadata = render_scene(asset_path, asset_root, output_root, hdri_paths, view_index, config, rng)
                rows[split].append(metadata)
                emit_progress("advance", scene=metadata["id"], split=split, failed=False)
            except Exception as error:
                record = {"asset": str(asset_path), "view": view_index, "error": repr(error)}
                errors.append(record)
                emit_progress(
                    "advance",
                    scene=f"{asset_path.name}:v{view_index:02d}",
                    split=split,
                    failed=True,
                    error=repr(error),
                )
    worker_root = output_root / "render_workers" / f"worker_{worker_index:03d}"
    for split, split_rows in rows.items():
        write_jsonl(worker_root / f"{split}.jsonl", split_rows)
    write_jsonl(worker_root / "render_errors.jsonl", errors)
    print(
        f"Done: {len(rows['train'])} train, {len(rows['validation'])} validation, "
        f"{len(rows['test'])} test, {len(errors)} failed",
        flush=True,
    )


def emit_progress(event: str, **payload) -> None:
    print(PROGRESS_PREFIX + json.dumps({"event": event, **payload}), flush=True)


def scan_files(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def assign_splits(assets: list[Path], config: dict) -> dict[Path, str]:
    fractions = [float(config["train_fraction"]), float(config["validation_fraction"]), float(config["test_fraction"])]
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError("render train/validation/test fraction 之和必须为 1")
    shuffled = list(assets)
    random.Random(int(config["split_seed"])).shuffle(shuffled)
    validation_count = round(len(assets) * fractions[1])
    test_count = round(len(assets) * fractions[2])
    validation = set(shuffled[:validation_count])
    test = set(shuffled[validation_count : validation_count + test_count])
    return {asset: "validation" if asset in validation else "test" if asset in test else "train" for asset in assets}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.worlds, bpy.data.meshes, bpy.data.armatures, bpy.data.curves,
        bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_asset(path: Path):
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        (bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj)(filepath=str(path))
    elif suffix == ".blend":
        with bpy.data.libraries.load(str(path), link=False) as (source, target):
            target.objects = source.objects
        for obj in target.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in [item for item in imported if item.type in {"LIGHT", "CAMERA"}]:
        bpy.data.objects.remove(obj, do_unlink=True)
    imported = [obj for obj in imported if obj.name in bpy.data.objects]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError("asset 不包含 mesh")
    return imported, meshes


def world_bounds(meshes):
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def normalize_asset(objects, meshes, size: float) -> float:
    root = bpy.data.objects.new("AssetRoot", None)
    bpy.context.collection.objects.link(root)
    for obj in objects:
        if obj.parent is None:
            matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = matrix
    minimum, maximum = world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    extent = max(maximum[index] - minimum[index] for index in range(3))
    if extent <= 1e-8:
        raise ValueError("asset bounding box 为零")
    scale = size / extent
    root.scale = (scale,) * 3
    root.location = -center * scale
    bpy.context.view_layer.update()
    minimum, _ = world_bounds(meshes)
    root.location.z -= minimum.z
    bpy.context.view_layer.update()
    return scale


def add_stage(config: dict) -> None:
    size = float(config["ground_size"])
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    material = bpy.data.materials.new("StageMaterial")
    material.diffuse_color = tuple(config["stage_color"])
    bpy.context.object.data.materials.append(material)
    if config["add_back_wall"]:
        bpy.ops.mesh.primitive_plane_add(
            size=size, location=(0, size * 0.25, size * 0.25), rotation=(math.pi / 2, 0, 0)
        )
        bpy.context.object.data.materials.append(material)


def look_at(obj, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_camera(config: dict, rng: random.Random):
    distance = float(config["camera_distance"])
    yaw = math.radians(rng.uniform(*config["camera_yaw_degrees"]))
    pitch = math.radians(rng.uniform(*config["camera_pitch_degrees"]))
    target = Vector((0, 0, float(config["camera_target_height"])))
    location = target + Vector(
        (distance * math.sin(yaw) * math.cos(pitch), -distance * math.cos(yaw) * math.cos(pitch), distance * math.sin(pitch))
    )
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    camera_data.lens = float(config["camera_focal_length"])
    look_at(camera, target)
    bpy.context.scene.camera = camera
    return camera, target


def camera_to_world(camera, target, position):
    forward = (target - camera.location).normalized()
    right = forward.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(forward).normalized()
    return target + right * position[0] + forward * position[1] + up * position[2]


def configure_world(hdri: Path | None):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 1.0
    if hdri:
        environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(str(hdri), check_existing=True)
        world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    else:
        background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return background


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
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.use_persistent_data = bool(config.get("persistent_data", True))
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


def create_fixtures(camera, target, config: dict, rng: random.Random) -> list[dict]:
    fixtures = []
    ranges = config["in_scene_position_ranges"]
    for index in range(int(config["in_scene_lights_per_scene"])):
        canonical = [rng.uniform(*ranges[axis]) for axis in ("x", "y", "z")]
        world_position = camera_to_world(camera, target, canonical)
        mesh, material = add_fixture(
            f"Fixture{index:03d}", world_position, float(config["in_scene_fixture_size"])
        )
        fixtures.append(
            {
                "mesh": mesh,
                "material": material,
                "position": canonical,
                "world_position": world_position,
            }
        )
    return fixtures


def remove_light(light) -> None:
    data = light.data
    bpy.data.objects.remove(light, do_unlink=True)
    if data.users == 0:
        bpy.data.lights.remove(data)


def render_exr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_fixture_mask(path: Path, fixture, material, background, mask_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    hidden = {obj: obj.hide_render for obj in bpy.data.objects if obj.type == "MESH"}
    original_strength = background.inputs["Strength"].default_value
    original_format = scene.render.image_settings.file_format
    original_mode = scene.render.image_settings.color_mode
    original_depth = scene.render.image_settings.color_depth
    original_transparent = scene.render.film_transparent
    original_samples = scene.cycles.samples if scene.render.engine == "CYCLES" else None
    original_denoising = scene.cycles.use_denoising if scene.render.engine == "CYCLES" else None
    try:
        for obj in hidden:
            obj.hide_render = obj != fixture
        background.inputs["Strength"].default_value = 0.0
        set_fixture_emission(material, (1.0, 1.0, 1.0, 1.0), 1.0)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.color_depth = "8"
        scene.render.film_transparent = False
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = mask_samples
            scene.cycles.use_denoising = False
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    finally:
        set_fixture_emission(material, (1.0, 1.0, 1.0, 1.0), 0.0)
        background.inputs["Strength"].default_value = original_strength
        scene.render.image_settings.file_format = original_format
        scene.render.image_settings.color_mode = original_mode
        scene.render.image_settings.color_depth = original_depth
        scene.render.film_transparent = original_transparent
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = original_samples
            scene.cycles.use_denoising = original_denoising
        for obj, was_hidden in hidden.items():
            obj.hide_render = was_hidden


def render_scene(asset_path, asset_root, output_root, hdris, view_index, config, rng):
    sid = make_scene_id(asset_path, asset_root, view_index)
    directory = output_root / "components" / sid
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file() and not config["overwrite"]:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    clear_scene()
    objects, meshes = import_asset(asset_path)
    asset_scale = normalize_asset(objects, meshes, float(config["canonical_size"]))
    add_stage(config)
    camera, target = add_camera(config, rng)
    fixtures = create_fixtures(camera, target, config, rng)
    hdri = rng.choice(hdris) if hdris else None
    background = configure_world(hdri)
    ambient_path = directory / "ambient.exr"
    emit_progress("stage", scene=sid, stage="ambient")
    render_exr(ambient_path)
    background.inputs["Strength"].default_value = 0
    dark_path = directory / "dark.exr"
    emit_progress("stage", scene=sid, stage="dark")
    render_exr(dark_path)
    in_scene_components = []
    fixture_energy = float(config["in_scene_light_energy"])
    fixture_radius = float(config["in_scene_light_radius"])
    fixture_light = (
        add_light("FixtureLight", "POINT", fixtures[0]["world_position"], 0.0, fixture_radius)
        if fixtures else None
    )
    try:
        for index, fixture in enumerate(fixtures):
            mask_path = directory / "in_scene_lights" / f"fixture_{index:03d}_mask.png"
            emit_progress("stage", scene=sid, stage=f"fixture-mask {index + 1}/{len(fixtures)}")
            render_fixture_mask(
                mask_path,
                fixture["mesh"],
                fixture["material"],
                background,
                int(config.get("mask_samples", 1)),
            )
            set_fixture_emission(
                fixture["material"],
                (1.0, 1.0, 1.0, 1.0),
                float(config["in_scene_emission_strength"]),
            )
            fixture_light.location = fixture["world_position"]
            fixture_light.data.energy = fixture_energy
            bpy.context.view_layer.update()
            component_path = directory / "in_scene_lights" / f"fixture_{index:03d}_on.exr"
            emit_progress("stage", scene=sid, stage=f"in-scene-light {index + 1}/{len(fixtures)}")
            render_exr(component_path)
            fixture_light.data.energy = 0.0
            set_fixture_emission(fixture["material"], (1.0, 1.0, 1.0, 1.0), 0.0)
            in_scene_components.append(
                {
                    "path": component_path.relative_to(output_root).as_posix(),
                    "mask": mask_path.relative_to(output_root).as_posix(),
                    "position": fixture["position"],
                    "renderer_position": list(fixture["world_position"]),
                    "base_energy": fixture_energy,
                    "diffuse": fixture_radius,
                    "fixture_size": float(config["in_scene_fixture_size"]),
                }
            )
    finally:
        if fixture_light is not None:
            remove_light(fixture_light)
    energy = float(config["point_light_base_energy"])
    point_components = []
    point_light_count = int(config["point_lights_per_scene"])
    point_light = add_light("PointLight", "POINT", (0, 0, 0), energy, 0.0) if point_light_count else None
    try:
        for index in range(point_light_count):
            position = [rng.uniform(*config["light_position_ranges"][axis]) for axis in ("x", "y", "z")]
            radius = rng.uniform(*config["point_light_radius_range"])
            world_position = camera_to_world(camera, target, position)
            point_light.location = world_position
            point_light.data.shadow_soft_size = radius
            bpy.context.view_layer.update()
            path = directory / "point_lights" / f"light_{index:03d}.exr"
            emit_progress("stage", scene=sid, stage=f"point-light {index + 1}/{point_light_count}")
            render_exr(path)
            point_components.append(
                {"path": path.relative_to(output_root).as_posix(), "position": position,
                 "renderer_position": list(world_position), "base_energy": energy, "diffuse": radius}
            )
    finally:
        if point_light is not None:
            remove_light(point_light)
    diffuse_components = []
    world_position = camera_to_world(camera, target, config["diffuse_light_position"])
    spreads = config["diffuse_sizes"]
    area_light = add_light("AreaLight", "AREA", world_position, energy, float(spreads[0])) if spreads else None
    try:
        if area_light is not None:
            look_at(area_light, target)
        for index, spread in enumerate(spreads):
            area_light.data.size = float(spread)
            bpy.context.view_layer.update()
            path = directory / "diffuse" / f"spread_{index:02d}.exr"
            emit_progress("stage", scene=sid, stage=f"diffuse {index + 1}/{len(spreads)}")
            render_exr(path)
            diffuse_components.append(
                {"path": path.relative_to(output_root).as_posix(),
                 "level": index / max(len(spreads) - 1, 1), "size": float(spread)}
            )
    finally:
        if area_light is not None:
            remove_light(area_light)
    metadata = {
        "id": sid,
        "asset_uid": asset_path.stem,
        "asset": asset_path.relative_to(asset_root).as_posix(),
        "asset_scale": asset_scale,
        "hdri": str(hdri) if hdri else None,
        "ambient": ambient_path.relative_to(output_root).as_posix(),
        "dark": dark_path.relative_to(output_root).as_posix(),
        "point_lights": point_components,
        "diffuse": diffuse_components,
        "in_scene_lights": in_scene_components,
        "camera": {"location": list(camera.location), "rotation_euler": list(camera.rotation_euler),
                   "focal_length": camera.data.lens, "target": list(target)},
        "canonical": {"origin": list(target), "asset_size": float(config["canonical_size"]),
                      "position_axes": "x=right,y=camera-forward,z=up"},
    }
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def make_scene_id(asset: Path, root: Path, view: int) -> str:
    relative = asset.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{asset.stem}_{digest}_v{view:02d}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


if __name__ == "__main__":
    if bpy is None:
        launch_blender(parse_outer_args().config)
    else:
        blender_main()
