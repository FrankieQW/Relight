# datasimplecompose

把**单个 3D 资产**放进一个**程序化「简单舞台」**里，渲染 TokenLight 需要的线性光照分量，
并可选择输出**内禀通道**（albedo / irradiance / normal）与**相机坐标系**（camera_position / image_plane）。

不需要任何真实场景（Bistro 之类）数据：地面、背墙、相机、点光源、面光源、画面内灯具全部按配置生成。
本目录是独立项目，输出字段与 `datacompose` 的渲染器保持一致。

## 依赖

- 外层 Python：`PyYAML`（读配置）、`tqdm`（进度条）
- `Blender 4.5`（`engine: CYCLES`，建议 NVIDIA GPU + CUDA）
- Blender 内：`numpy`（读 EXR、推导 irradiance）

## 用法

```bash
python render_assets.py --config configs/default.yaml
```

- 普通 Python 启动：读 YAML → 写 `render_config.*.snapshot.json` →
  `blender --background --python render_assets.py -- --runtime-config <snapshot>`，
  按 `gpu_ids × workers_per_gpu` 起多个 worker（`CUDA_VISIBLE_DEVICES` 分卡），父进程合并 manifest。
- 也可在 Blender 内直接跑（走 `blender_main()`）。

## 配置

见 `configs/default.yaml`：

| 组 | 关键项 |
|---|---|
| `paths` | `object_root`（递归扫描 `.glb/.gltf/.fbx/.obj/.blend`）、`hdri_root`（可 `null`）、`render_output_root` |
| 并行/渲染 | `blender_executable`、`gpu_ids`、`workers_per_gpu`、`threads_per_worker`、`resolution`、`engine`、`samples`、`denoise`、`mask_samples`、`device`、`compute_device_type`、`require_gpu`、`persistent_data` |
| 舞台/光 | `canonical_size`、`ground_size`、`stage_color`、`add_back_wall`、`camera_*`、`point_lights_per_scene`、`point_light_base_energy`、`point_light_radius_range`、`light_position_ranges`、`diffuse_sizes`、`diffuse_light_position`、`in_scene_*` |
| 采样 | `views_per_asset`、`train/validation/test_fraction`（和为 1）、`split_seed`、`asset_limit`、`overwrite` |
| 内禀通道 | `passes.*`（见下） |

## 相机坐标系

每个 `metadata.json` 的 `camera` 块都带完整针孔参数：

```json
"camera": {
  "location": [...], "rotation_euler": [...], "target": [...],
  "focal_length": 50.0, "coordinate_space": "blender_world_meter",
  "shift_x": 0.0, "shift_y": 0.0,
  "sensor_width": 36.0, "sensor_fit": "AUTO",
  "angle_x": 0.69, "angle_y": 0.69,
  "resolution": {"width": 960, "height": 960},
  "intrinsics": {"width": 960, "height": 960, "fx": 691.6, "fy": 691.6, "cx": 480.0, "cy": 480.0}
}
```

`canonical` 固定为 `x=right, y=camera-forward, z=up`，原点在 `camera.target`，单位米。

渲染器最后用 `coordinates.convert_row()` 给每个 `point_lights` / `in_scene_lights` 元素补两个字段
（与 datacompose 的 manifest 完全同一套实现）：

- `camera_position`：OpenCV 相机系 `[right, down, forward]`，即 `[x, -z, y + distance]`
- `image_plane`：`{x, y, u, v, depth}`，`u/v` 为左上角原点的像素坐标，`depth = camera_position[2]`

并在行上写 `camera_position_convention` 说明该批数据用的约定与内参。

## 内禀通道（三张图）

`render.passes.enabled: true` 时，每个样本额外的 `passes/` 目录：

```text
passes/
├── albedo.exr          # Cycles Diffuse Color（无光照反照率）
├── normal_camera.exr   # 相机空间单位法线（OpenCV：x右 y下 z前）
├── valid_mask.png      # albedo >= eps 且法线非零的像素
├── material_audit.json # 贴图 / Base Color 连接情况（含缺失贴图清单）
└── irradiance/         # (DiffuseDirect + DiffuseIndirect) / albedo
    ├── ambient.exr
    ├── light_000.exr ...
    └── fixture_000.exr
```

定义（与 datacompose 相同）：

```text
albedo(p)        = Cycles Diffuse Color(p)
irradiance_L(p)  = (DiffuseDirect_L + DiffuseIndirect_L)(p) / albedo(p)   # E/pi
normal_camera(p) = R_cam^T · N_world(p)
valid(p)         = albedo(p) >= eps 且 ||N_world(p)|| > 0
```

要点：

- `albedo` / `normal_world` 用 `pass_samples`（默认 1）单独渲染一次参考帧，`film_transparent` 打开，
  因此与光照渲染无关，`valid_mask` 在背景处为 0。
- `irradiance` **不额外渲染**：与分量图同一次渲染顺带写出 `Diffuse Color/Direct/Indirect`，
  再在 Python 里相除，原始 pass 用完即删。
- `require_textures: true` 时，贴图缺失或 Base Color 未接线会让该样本直接报错（默认只记录在
  `material_audit.json` 与 `quality` 里）。
- 关闭 `passes.enabled` 时，输出与引入该功能前**完全一致**（不产生 `passes/` 目录）。

## 输出

```text
<render_output_root>/
├── components/<scene_id>/
│   ├── ambient.exr  dark.exr
│   ├── point_lights/light_XXX.exr
│   ├── diffuse/spread_XX.exr
│   ├── in_scene_lights/fixture_XXX_on.exr + fixture_XXX_mask.png
│   ├── passes/…                # 仅 passes.enabled=true
│   └── metadata.json
├── render_workers/worker_XXX/{train,validation,test}.jsonl
├── manifests/{train,validation,test}.jsonl
└── render_errors.jsonl
```

`scene_id = <资产名>_<相对路径 sha1 前 10 位>_v<视角序号>`。

## 排查渲染错误

单个样本的异常不会中断整批，但**默认只把 `repr(error)` 写进 `render_errors.jsonl`**，看不到行号。所以：

- 每个 worker 的**第一条异常**会把完整 traceback 写到
  `<render_output_root>/render_first_error.worker_XXX.txt`，同时打到 stdout；
- `render_errors.jsonl` 的每条记录除 `asset/view/error` 外还带 `traceback` 字段；
- 想立刻在终端看到堆栈：`fail_fast: true` + `asset_limit: 1` + `gpu_ids: [0]` + `workers_per_gpu: 1`，
  只跑 1 个资产、1 个 worker，首个异常直接抛出，父进程会打印该 worker 的末尾输出。

```bash
# 最小复现
python render_assets.py --config configs/debug.yaml   # 上面四个键的临时配置
```

## 说明

- 每个样本是**单物体**，没有 `base_scene_id` / `composition`（那是场景合成的字段）。
- `overwrite: false` 时已有的 `metadata.json` 直接复用，便于断点续跑。
- 多卡：`gpu_ids: [0,1]` + `workers_per_gpu: 2` → 4 个 worker，资产按索引取模分片，父进程合并。
