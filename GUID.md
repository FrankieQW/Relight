# datacompose 数据合成指南

本目录是 LightConstruction 的**纯场景数据合成**代码集合（可独立复制运行的 compose 项目）：在已有场景
（Bistro 等 FBX）里规划相机、施加可控光照，渲染出 TokenLight 需要的线性光照分量与 manifest。
**不再插入 Objaverse 物体**，也没有物体/场景的语义配对；一个样本是 `(scene, camera pose, lighting)`。
坐标约定与本目录内 `transfer.py` 完全一致，并在此基础上补齐了**相机坐标系下的图像平面坐标与深度**。

- 参考实现：本目录 `transfer.py`（纯标准库，作为坐标约定的权威脚本）。
- 模块化实现：`datacompose/coordinates.py`（本包内，供流水线和测试直接导入，并多写 `image_plane`）。
- 一致性保证：`datacompose/tests/test_coordinates.py` 用同一份 manifest 同时跑两套实现，断言
  `camera_position` 逐位相等。

---

## 1. 目录结构

```text
datacompose/
├── GUID.md                     本文档
├── README.md                   独立项目快速上手
├── __init__.py                 __version__
├── __main__.py                 python -m datacompose
├── _launcher.py                脚本启动器：包名注册 + 缺省 --config
├── cli.py                      M1–M4 统一入口（依赖惰性导入）
├── prepare_objects.py          python prepare_objects.py        → prepare-objects
├── prepare_scenes.py           python prepare_scenes.py         → prepare-scenes
├── annotate_construction.py    python annotate_construction.py  → annotate-construction
├── prepare_geometry.py         python prepare_geometry.py       → prepare-geometry
├── build_render_jobs.py        python build_render_jobs.py      → build-render-jobs
├── render_dataset.py           python render_dataset.py         → render
├── clear_render_partial.py     python clear_render_partial.py   → clear-render-partial
├── plan_scene_cameras.py       python plan_scene_cameras.py     → plan-scene-cameras
├── config.py                   YAML 配置与项目根解析
├── coordinates.py              相机坐标 / 图像平面 / 深度（transfer.py 的模块版 + 扩展）
├── intrinsic.py                物理表征通道推导（albedo / irradiance / normal，纯 NumPy）
├── io_utils.py                 原子写、digest、JSONL
├── schemas.py                  Pydantic 数据协议（场景 + 相机 job）
├── scene_semantics.py          类别归一化与别名
├── scene_cameras.py            场景内无碰撞相机采样（AABB 宽相 + 视野无穿插判据）
├── scene_extract.py            prepare-scenes（调用 blender/entry.py）
├── render_scenes.py            render（多 GPU worker 调度 blender/render_scene.py）
├── render_recovery.py          渲染 partial 的清理与恢复
├── transfer.py                 坐标约定权威参考实现（纯标准库）
├── configs/
│   ├── default.yaml            项目配置（project.root=.. 指向本目录）
│   ├── category_aliases.yaml
│   └── scene_overrides.yaml
├── blender/
│   ├── entry.py                FBX 导入、归一化、稳定 ID、scene fragment
│   └── render_scene.py         相机位姿精确复核、光照、分量渲染、metadata
├── tokenlight/
│   ├── render_assets.py        Blender 侧渲染工具（供 render_scene.py 动态加载）
│   ├── build_manifests.py      components → train/validation/test.jsonl（按 scene 切分）
│   └── validate_components.py  分量完整性与坐标约定校验
├── scripts/
│   └── remove_legacy_object_pipeline.sh   清理 object 时代残留文件
├── tools/
│   └── bench_scene_cameras.py  相机搜索成本基准
├── docs/
│   └── scene_only_render_feasibility.md   纯场景合成可行性方案
└── tests/
    ├── test_coordinates.py
    ├── test_intrinsic_passes.py
    ├── test_manifest_camera.py
    ├── test_render_recovery.py
    ├── test_launcher.py
    └── test_scene_cameras.py
```

本目录是**独立可复制的 compose 项目**：只依赖自身代码与本目录下的 `configs/`、`transfer.py`，
不引用上级仓库的 `src/`、`scripts/`、`Lumina-T2X/`。场景数据（Bistro 等 FBX）与训练侧
由外部提供，通过 `--config` 的数据路径接入；原 `stage2`（训练编排）已移出本项目。

---

## 2. 数据流

```text
data/scene/scene.json ── plan-scene-cameras ─→ scene_camera_jobs.jsonl
                                                      │
                                      render（Blender：位姿复核 + 渲染）
                                                      │
              components/<job_id>/{ambient,dark,point_lights,diffuse,in_scene_lights}.exr
              components/<job_id>/metadata.json      ← canonical + world 坐标在这里落盘
                                                      │
                          build_manifests → validate_components
                                                      │
              manifests/{train,validation,test}.jsonl  ← 写入 camera_position / image_plane
```

一个 render job 是 `(scene, camera pose, lighting profile, seed)`，与物体无关。每个 job 都从只读 base
`.blend` 重新打开，渲染完回滚，不把上一个样本的状态带进下一个。

相机位姿由 `scene_cameras.py` 规划（AABB 宽相），再由 `blender/render_scene.py` 用真实三角面复核：
穿过几何体、贴脸、只对着一面墙、深度过平都会让这个 job 直接失败，而不是产出一张坏图。

渲染产物里的 `position` 是 canonical 坐标系；训练需要的 OpenCV 相机坐标由 `build_manifests.py` 在
生成本文件时写入 `camera_position`，因此 **manifest 才是训练该读的那份数据**，`components/*/metadata.json`
保持为原始渲染记录。

---

## 3. 坐标系

渲染链路上同时存在四个坐标系。理解它们之间的换算，是读懂 manifest 里灯光坐标的前提。

| 名称 | 轴向 | 原点 | 单位 | 谁写入 |
|---|---|---|---|---|
| Blender world | 右手，`+Z` 向上 | 场景世界原点 | 米 | Blender |
| canonical | `x=right, y=camera-forward, z=up` | `camera.target` | 米 | `render_scene.py` |
| camera（OpenCV） | `X=right, Y=down, Z=forward` | `camera.location` | 米 | `transfer.py` / `coordinates.py` |
| image plane | `u` 向右、`v` 向下 | 图像左上角像素 | 像素 | `coordinates.py` |

### 3.1 canonical 坐标系

`canonical.position_axes` 固定为 `x=right,y=camera-forward,z=up`。一个光源的 canonical 位置
`[x, y, z]` 表示：

- `x`：沿相机右方向、相对 `camera.target` 的偏移；
- `y`：沿相机前方向、相对 `camera.target` 的偏移（朝向场景为正）；
- `z`：沿相机上方向、相对 `camera.target` 的偏移。

相机基向量由 target-facing、无 roll 的约定给出：

```text
forward = normalize(target - location)
right   = normalize(cross(forward, +Z_world))
up      = normalize(cross(right, forward))
distance = |target - location|
```

`render_scene.py` 用 `_world_to_canonical()` 把灯光世界坐标写进 `position`，随后 `transfer.py`
或 `coordinates.py` 负责换算。渲染器还会把 `rotation_euler` 写进 manifest，转换器据此校验保存的相机
确实是“对准 target、无 roll”，避免用错公式却不报错。

### 3.2 camera 坐标系（`camera_position`）

OpenCV 相机坐标系规定 `X` 向右、`Y` 向下、`Z` 向前，原点在相机光心。因为 canonical 的 `z` 是**向上**，
换算时竖直分量取反；因为 canonical 的原点在 target、相机原点在光心，前向分量要加上 `distance`：

```text
camera_position = [x, -z, y + distance]

X = x
Y = -z
Z = y + distance
```

这正是 `transfer.py` 里的那一行：

```python
converted = [x, -z, y + distance]
```

例：相机在 `[0, -10, 0]`、target 在原点、光源 canonical 位置 `[2, 3, 1]`，则
`distance = 10`，`camera_position = [2, -1, 13]`。

### 3.3 图像平面坐标与深度

透视投影使用的就是相机坐标的三个分量：

```text
x_normalized = X / Z
y_normalized = Y / Z

u = fx * X / Z + cx
v = fy * Y / Z + cy
```

其中 `u`、`v` 是像素坐标（浮点），原点在图像**左上角**，`u` 向右、`v` 向下；`depth = Z`。内参由渲染
时的相机直接给出，并写进 manifest 的 `camera.intrinsics`：

```text
fx = (width  / 2) / tan(angle_x / 2)
fy = (height / 2) / tan(angle_y / 2)
cx = width  / 2 - width  * shift_x
cy = height / 2 + height * shift_y
```

`angle_x`、`angle_y` 是 Blender 的相机视场角（弧度），`shift_x`、`shift_y` 是 Blender 的镜头平移。
平移只移动画幅、不旋转相机，所以它体现在**主点偏移**上，而不是焦距上。由上式可验证：`shift_x > 0`
时 `cx` 变小（画面内容左移），`shift_y > 0` 时 `cy` 变大（画面内容下移），与 Blender 的
`world_to_camera_view()` 一致。

同一组内参下，`u`、`v` 与 `x_normalized`、`y_normalized` 互为线性变换：

```text
x_normalized = (u - cx) / fx
y_normalized = (v - cy) / fy
X = x_normalized * Z
Y = y_normalized * Z
```

所以“图像平面坐标 + 深度”与“相机三维坐标”是**等价且无损**的两种写法：前者是二维位置加一个标量，
后者是三个分量。`coordinates.py` 两份都写，`image_plane` 里同时给出像素坐标、`[0,1]` 归一化坐标和
`x`/`y` 归一化图像坐标，便于不同下游直接取用。

### 3.4 深度到底是谁到谁的距离

`depth` 的定义是：

> 光源点相对于相机光心，沿相机前方向（光轴）的**有符号**投影长度。

它有三个容易混淆的近邻，区别如下：

| 量 | 公式 | 说明 |
|---|---|---|
| `depth`（本文） | `y + distance` | 沿光轴的轴向距离，透视投影分母 |
| 到相机的欧氏距离 | `sqrt(X² + Y² + Z²)` | 偏轴时会大于 `depth` |
| 到成像平面的距离 | `|Z - f|` | 需要焦距 `f` 才有定义 |

例：上面的 `[2, -1, 13]`，`depth = 13`，而到相机光心的直线距离是
`sqrt(2² + 1² + 13²) ≈ 13.19`。若虚拟成像平面放在 `Z = f = 50 mm`，点到该平面的轴向距离才是
`Z - f`；本流水线不产出这个量，也不使用它。

`depth` 可以为负或零：光源在相机后方时为负（`image_plane.in_front = false`），与光心同深度时为 0，
此时投影无定义，`coordinates.py` 会抛错而不是写出 `inf`。

### 3.5 归一化策略

`transfer.py` 与 `coordinates.py` 都**不做跨场景归一化**，`depth` 就是米制的绝对深度。manifest 的
`camera_position_convention.units` 沿用 `canonical.units`（当前渲染器写 `meter`）。

不同 scene 的绝对尺度本来就可能不同（相机距离、物体尺寸、房间大小都不一样）。训练侧若需要可比的深度，
建议在数据层显式选择一种口径，并把所用口径写进实验配置：

| 口径 | 公式 | 适用 |
|---|---|---|
| 绝对深度 | `Z` | 需要真实米制尺度、或做物理一致的重新打光 |
| 相对相机距离 | `Z / distance` | 构图相近、希望 target 深度恒为 1 |
| 除以场景尺度 | `Z / S` | 每个 scene 有明确的规范化尺度 `S`，保留场景内相对关系 |
| 视锥映射 | `(Z - near) / (far - near)` | 需要深度图样式输出，会丢失绝对单位 |

不建议对每张图单独做 min–max：同一深度值在不同图上会被映射成不同数值，模型无法学到跨场景的深度关系。

---

## 4. manifest 字段

### 4.1 相机块

```json
"camera": {
  "location": [0.0, -10.0, 0.0],
  "target": [0.0, 0.0, 0.0],
  "rotation_euler": [1.5707963267948966, 0.0, 0.0],
  "focal_length": 18.0,
  "shift_x": 0.0,
  "shift_y": 0.0,
  "sensor_width": 36.0,
  "sensor_fit": "AUTO",
  "angle_x": 1.5707963267948966,
  "angle_y": 1.5707963267948966,
  "resolution": {"width": 1024, "height": 1024},
  "intrinsics": {"width": 1024, "height": 1024, "fx": 512.0, "fy": 512.0, "cx": 512.0, "cy": 512.0}
}
```

`intrinsics` 是权威值；`resolution` + `angle_x/angle_y` + `shift_*` 是可重算的冗余信息，便于老的
manifest 或外部工具自行恢复。`focal_length` 与 `sensor_width` 只是 Blender 侧的原始参数。

### 4.2 光源块

```json
{
  "position": [2.0, 3.0, 1.0],
  "renderer_position": [2.0, 3.0, 1.0],
  "camera_position": [2.0, -1.0, 13.0],
  "image_plane": {
    "u": 590.7692307692307,
    "v": 472.61538461538464,
    "x_ndc": 0.5769230769230769,
    "y_ndc": 0.46153846153846156,
    "x": 0.15384615384615385,
    "y": -0.07692307692307693,
    "depth": 13.0,
    "in_front": true,
    "width": 1024,
    "height": 1024,
    "projection": "pinhole"
  }
}
```

- `position`：canonical 坐标，由渲染器写，转换器只读不改。
- `renderer_position`：Blender 世界坐标。转换器用它交叉校验换算结果，不一致即报错。
- `camera_position`：OpenCV 相机坐标，训练代码读这个字段。`build_manifests.py` 在生成
  `manifests/*.jsonl` 时逐行写入，因此出厂 manifest 已带训练可用的 xyz。
- `image_plane`：像素位置 + 深度，以及可脱离内参使用的 `x`/`y` 归一化坐标；内参缺失时只写深度与
  归一化坐标，不写像素。

### 4.3 文件级约定

```json
"camera_position_convention": {
  "coordinate_space": "opencv_camera",
  "origin": "camera.location",
  "axes": "x=right,y=down,z=forward",
  "units": "meter",
  "depth": "signed_distance_along_camera_forward_axis",
  "image_plane": "pinhole_u_v_pixels_top_left_origin",
  "position_formula": "camera_position = [canonical_x, -canonical_z, canonical_y + distance]",
  "intrinsics": {"width": 1024, "height": 1024, "fx": 512.0, "fy": 512.0, "cx": 512.0, "cy": 512.0}
}
```

`validate_components.py` 会检查 `canonical.position_axes` 是否等于约定值，要求每个
`point_lights` 与 `in_scene_lights` 都能换算出一致的相机坐标（`coordinates.convert_row()` 内部交叉
校验 `position` 与 `renderer_position`），缺失或不一致的 manifest 会被数据门直接拒绝；同时它检查
分量文件存在且非空、`camera` 带内参、以及同一个 `scene_id` 不跨 split。

`build_manifests.py` 会在 `dataset_release.json` 写入
`camera_position: {axes, depth, rows_with_image_plane_pixels}`，用于确认这一批数据确实带了训练坐标、
以及有多少行同时具备像素坐标。

---

## 5. 坐标换算 API

```python
from datacompose import coordinates
import json

row = json.loads(next(open("data/tokenlight_dataset/train.jsonl", encoding="utf-8")))

frame = coordinates.CameraFrame.from_row(row)
frame.location, frame.target, frame.distance
frame.right, frame.up, frame.forward

# canonical <-> camera
frame.canonical_to_camera([2.0, 3.0, 1.0])   # [2.0, -1.0, 13.0]
frame.camera_to_canonical([2.0, -1.0, 13.0]) # [2.0, 3.0, 1.0]

# world <-> camera
frame.world_to_camera([2.0, 3.0, 1.0])
frame.camera_to_world([2.0, -1.0, 13.0])

# 图像平面
intrinsics = coordinates.Intrinsics.from_camera(row["camera"])
intrinsics.project([2.0, -1.0, 13.0])          # (u, v)
intrinsics.unproject(x_normalized, y_normalized, depth)  # [X, Y, Z]
```

`manifests/*.jsonl` 由 `build_manifests.py` 生成时就已写好 `camera_position`（以及能算就写的
`image_plane`），所以训练侧不需要额外步骤。下面的命令用于两种情况：修复历史 manifest（生成时还没有这一步），
或对 `components/*/metadata.json` 这类非 JSONL 的原始产物做独立核对。

把一整个目录的 manifest 补齐相机坐标与图像平面：

```bash
python -m datacompose.coordinates \
  data/tokenlight_dataset/manifests \
  data/tokenlight_dataset_camera

# 渲染早于 intrinsics 字段的旧 manifest，显式给出分辨率：
python -m datacompose.coordinates old_manifests new_manifests --resolution 960

# 只写深度、不要求像素坐标：
python -m datacompose.coordinates old_manifests new_manifests --allow-missing-intrinsics
```

行为约定：

- 输入输出目录必须不同，默认拒绝覆盖已有输出文件；
- 先在输出目录内的临时目录写完全部文件再逐个发布，失败不会留下半个 manifest；
- `camera_position` 总是写；`image_plane` 在内参可得时写；
- 默认要求内参，缺失时报错并提示 `--resolution`，不会默默写出错误的像素坐标。

---

## 6. 运行

推荐用**同名脚本**直接跑，不需要 `python -m`，也不需要站在特定目录（脚本自己解析包位置）：

```bash
conda activate <env>
cd <本项目目录>              # 包含 prepare_scenes.py 等脚本的目录

python prepare_scenes.py       --blender-bin /opt/blender-4.5/blender
python plan_scene_cameras.py   --cameras-per-scene 32
python render_dataset.py       --blender-bin /opt/blender-4.5/blender
python clear_render_partial.py --job-id <job_id>
```

脚本与 CLI 子命令一一对应，参数完全一致：

| 脚本 | 等价 CLI |
|---|---|
| `prepare_scenes.py` | `python -m datacompose.cli prepare-scenes` |
| `plan_scene_cameras.py` | `... plan-scene-cameras` |
| `render_dataset.py` | `... render` |
| `clear_render_partial.py` | `... clear-render-partial` |

脚本的实现只有三行（`from _launcher import run` → `run("<command>")`），`_launcher.py` 负责两件事：

1. **把本目录按规范名 `datacompose` 注册进 `sys.modules`**，所以项目目录被改名（例如
   `datacompose-sceneonly`）或从任意工作目录启动都能 import；`blender/entry.py` 在 Blender 内部
   也做同样的注册。
2. **缺省 `--config`**：命令行没给时用环境变量 `DATACOMPOSE_CONFIG`，再退回 `configs/default.yaml`。

```bash
export DATACOMPOSE_CONFIG=/data/frankie/datacompose-sceneonly/sceneonly.yaml
python prepare_scenes.py --blender-bin /opt/blender-4.5/blender   # 不必再写 --config
```

`python -m datacompose.cli <command>` 仍然保留（要求当前目录是包的上一级），适合被别的脚本调用。
`cli.py` 的第三方依赖是**按命令惰性导入**的：`--help` 与 `prepare-scenes` 不需要渲染/LLM 依赖。

Blender 侧的调用路径由 `scene_extract.py` 和 `render_scenes.py` 从**包自身位置**解析
（`datacompose/blender/*.py`），不依赖 `config.root`，因此可从任意工作目录启动 Blender。
数据与输出路径由 config 的 `paths`（相对 `project.root`）决定；`paths.scene_cache` 必须位于
`project.root` 之内，因为场景片段会以相对路径回写。

训练/编排（原 stage2，依赖 Lumina-T2X）已不在本独立 compose 项目内，请在本目录之外单独进行。

---

## 7. 验收

```bash
python -m unittest discover -s datacompose/tests -p 'test_*.py'
```

`tests/test_launcher.py` 覆盖脚本化运行方式：包目录被改名后仍能按 `datacompose` 导入、
`DATACOMPOSE_CONFIG` 与 `--config` 的优先级、以及 `plan-scene-cameras` 的端到端产出。
`tests/test_scene_cameras.py` 覆盖场景相机判据（相机在几何体内 / 近裁剪穿插 / 单一平面糊屏 /
深度跨度、确定性、与 `coordinates` 契约的一致性）。

坐标相关的检查集中在 `datacompose/tests/test_coordinates.py`：

- 同一 manifest 上，`datacompose.coordinates.convert_row()` 与本目录 `transfer.py` 的
  `convert_row()` 产出逐位相等的 `camera_position`；
- 两者对 `renderer_position` 不一致的 manifest 都报错；
- `canonical ↔ camera`、`world ↔ camera` 往返一致；
- `depth` 与欧氏距离、与“到成像平面距离”的差异被显式断言；
- `u/v` 与 `fx/fy/cx/cy` 的投影公式一致，`image_plane_to_camera()` 能无损还原相机坐标；
- 镜头平移只改变主点，不改变焦距；
- 目录转换是原子的，且默认拒绝覆盖。

---

## 8. 与旧路径的映射

| 旧位置 | 新位置 |
|---|---|
| `src/lightconstruction/*.py` | `datacompose/*.py` |
| `src/lightconstruction/cli.py` | `datacompose/cli.py` |
| `scripts/blender_entry.py` | `datacompose/blender/entry.py` |
| `scripts/blender_render_composition.py` | `datacompose/blender/render_scene.py`（已改为纯场景渲染） |
| `Lumina-T2X/tools/tokenlight_data/render_assets.py` | `datacompose/tokenlight/render_assets.py` |
| `Lumina-T2X/tools/tokenlight_data/build_manifests.py` | `datacompose/tokenlight/build_manifests.py` |
| `Lumina-T2X/tools/tokenlight_data/validate_components.py` | `datacompose/tokenlight/validate_components.py` |
| 仓库根目录 `configs/*.yaml` | `datacompose/configs/*.yaml` |
| 仓库根目录 `tests/*.py` | `datacompose/tests/*.py` |
| 仓库根目录 `transfer.py` | `datacompose/transfer.py` |
| `python -m lightconstruction.cli` | `python -m datacompose.cli` |
| `from lightconstruction.x import y` | `from datacompose.x import y` |

`transfer.py` 现位于本目录，作为坐标约定的参考实现；`datacompose/coordinates.py` 是它的模块化
版本，多写 `image_plane`，其余数值行为与它一致。

> 训练/编排相关的旧 `scripts/stage2_config.py`、`scripts/run_stage2.sh`、`stage2.env.example`
> 已移出本独立 compose 项目（本项目只做数据合成）。
>
> 纯场景改造同时移除了 object/标注/组合链路（`object_index`、`geometry_prepare`、`llm_annotate`、
> `render_jobs`、`render_compositions`、`blender/render_composition`）。若某个 checkout 里还有这些
> 文件残留，用 `bash scripts/remove_legacy_object_pipeline.sh --apply` 清理。

---

## 9. 常见问题

**`depth` 为什么不是“到成像平面的距离”？**
成像平面只有在给定焦距后才存在，而 `depth` 是相机坐标系里的 `Z` 分量，不依赖内参。透视投影
`u = fx * X / Z + cx` 用的正是这个 `Z`。

**为什么不同 scene 的 `depth` 数量级差很多？**
因为 `depth` 是米制绝对深度，`transfer.py` 与 `coordinates.py` 都不做归一化。需要可比性时按 3.5 节
显式选择口径。

**`camera_position[2]` 可以是负数吗？**
可以。负值表示光源在相机后方，`image_plane.in_front` 同时给出布尔标记；恰好为 0 时无法投影，转换器
会报错。

**换了分辨率或裁切，内参还成立吗？**
`intrinsics` 描述的是渲染时的整幅图像。若下游做了中心裁剪或缩放，必须同步对 `cx/cy` 做平移、对
`fx/fy` 做等比缩放，否则 `u/v` 会整体偏移。

**manifest 里没有 `intrinsics` 怎么办？**
说明它由更早版本的渲染器生成。用 `--resolution WIDTHxHEIGHT` 给出像素尺寸，转换器会从
`angle_x/angle_y` + `shift_x/shift_y` 重新算出内参；连视场角也没有时，只能得到深度与归一化坐标。

---

## 10. 物理表征通道（albedo / irradiance / normal）

这一节描述 `m4.render.passes` 打开的**内在物理通道**。该开关写在操作者传入的 `--config`
的 `m4.render` 下；datacompose 内的代码默认 `enabled: false`（见 `_render_config`），
**不依赖任何外部样例配置**。它们与现有的
ambient / dark / point / diffuse / fixture 分量并存，**默认关闭**（`passes.enabled: false`），
关闭时产物与既有数据逐字节一致。开关打开后，每个 job 额外产出 3 张物理表征图 + 材质审计 + 质量
统计，用于蒸馏训练的本征分解监督（训练侧改动不在本仓库范围内）。

### 10.1 精确定义

```text
albedo(p)        = Cycles Diffuse Color(p)                       # 线性、无光照、无高光
irradiance_L(p)  = (DiffuseDirect_L + DiffuseIndirect_L)(p) / max(albedo(p), eps)   # = E_L/π
normal_camera(p) = R_cam^T · N_world(p)                          # 单位向量，X 右 / Y 下 / Z 前
valid(p)         = albedo(p) ≥ eps   且   ||N_world(p)|| ≈ 1
eps 默认 1e-3
```

- `albedo` 直接取 Cycles 的 **Diffuse Color** pass，与光照无关，因此同一物体的 albedo 在所有光照
  分量下都相同，一次渲染即可复用。
- `irradiance_L` 是**推导值**：把该分量渲染出的 `DiffuseDirect + DiffuseIndirect` 除以 albedo。
  它等于余弦加权辐照度除以 π，即“同一光照下白色朗伯表面会反出的值”。推导**复用**了已有分量的
  渲染 pass，**不增加分量渲染次数**。
- `normal_camera` 用相机基向量投影 `R_cam = [right, up, forward]`，转到 OpenCV 相机系
  （X 右、Y **下**、Z 前），与 manifest 里的 `camera_position` 同一坐标系。`intrinsic.world_normals_to_camera`
  负责这个转换（Y 轴取负）。
- `valid_mask` 标出可用的像素：albedo 三通道都 ≥ eps **且** 背后确有几何（法线非零）。

> 也可以把 `irradiance_mode` 设为 `override`：用白色朗伯材质覆盖渲染、发光灯具除外，得到精确对照，
> 渲染时间约 ×2，**不默认开启**。

### 10.2 产出布局

```text
components/<job_id>/
└── passes/
    ├── albedo.exr              # 每 job 一张（Diffuse Color）
    ├── normal_camera.exr       # 每 job 一张（相机空间单位法线）
    ├── normal_world.exr        # 可选（save_normal_world: true）
    ├── valid_mask.png          # 每 job 一张，8-bit
    ├── material_audit.json     # 每 job 一份，见 10.4
    └── irradiance/
        ├── ambient.exr
        ├── point_000.exr … point_015.exr
        └── fixture_000.exr
```

`metadata.json` 新增 `passes` 块（路径相对 dataset 根）：

```json
"passes": {
  "definition": {"albedo": "cycles_diffuse_color", "irradiance": "diffuse_radiance_over_albedo",
                 "normal": "camera_space_unit_opencv", "eps": 0.001},
  "albedo": "components/<id>/passes/albedo.exr",
  "normal_camera": "components/<id>/passes/normal_camera.exr",
  "valid_mask": "components/<id>/passes/valid_mask.png",
  "material_audit": "components/<id>/passes/material_audit.json",
  "quality": {"base_color_linked_ratio": 0.98, "magenta_ratio": 0.0, "textured_pixel_ratio": 0.34},
  "irradiance": {"ambient": "...", "point_000": "...", "fixture_000": "..."}
}
```

`build_manifests.py` 逐行透传 `metadata.json`，不改写 `passes` 路径；`dataset_release.json` 增加
`intrinsic_passes: {enabled, rows_with_passes, definitions, quality_thresholds}` 汇总。

### 10.3 与 TokenLight 分量的配对方式

- `irradiance/ambient.exr` ↔ metadata 的 `ambient` 分量；
- `irradiance/point_000.exr` ↔ `point_lights[0]`，编号与 `point_lights/light_000.exr` 一致；
- `irradiance/fixture_000.exr` ↔ `in_scene_lights[0]`（fixture）。

因此对任意分量分量 L 存在恒等式（在 `valid` 像素上）：

```text
component_L(p) − dark(p)  ≈  albedo(p) × irradiance_L(p)
```

这也是 §7/§10.7 的验收依据：`albedo>0.1` 的像素上相对误差中位数应 < 5%。

**金属 / 玻璃缺口**：金属与玻璃的漫反射分量接近 0，`albedo≈0`。此时 `albedo × irradiance`
恒等式在该像素不成立，除法也会放大噪声，因此这些像素被 `valid_mask` 直接排除，而不是猜测。
`eps` 之下的像素不参与任何本征统计。

### 10.4 材质接入判定（三层）

#### 第一层：渲染前材质审计（决定性，`audit_materials()`）

在 Blender 内遍历 `bpy.data.materials`，对含 **Image Texture** 节点的材质记录：

```json
{"material": "...", "has_nodes": true, "principled": true,
 "base_color_linked": true, "base_color_image": "…_BaseColor.dds",
 "image_exists": true, "image_size": [2048, 2048],
 "normal_linked": true, "metallic_linked": true, "roughness_linked": true}
```

汇总：`materials / textured / base_color_linked / base_color_linked_ratio / missing_image_files / missing_examples`。

判定规则：

- `image_exists=false` 或 `image_size` 为 0×0 → **等同没接上**（`base_color_linked=false`）；
- 只统计“节点树里含 Image Texture 的材质”作为分母，纯色材质不冤枉成失败；
- `base_color_linked` 要求 Base Color 能沿节点链追溯到 Image Texture 且该图存在；
- `require_textures: true` 时，`base_color_linked_ratio < min_base_color_linked_ratio(0.9)`
  或 `missing_image_files > 0` → **该 job 直接失败**，错误信息附前 10 个失败材质名。

#### 第二层：渲染后 albedo 统计（自动，`intrinsic.albedo_quality()`）

| 指标 | 算法 | 阈值 |
|---|---|---|
| `magenta_ratio` | R>0.99 & G<0.01 & B>0.99 的像素占比（Blender 缺图渲染成品红） | `> 1e-4` 报错 |
| `textured_pixel_ratio` | 5×5 局部方差 > 1e-6 的像素占比 | `< 0.02` 且 `require_textures` 时报错，否则仅告警 |
| `unique_colors` | 量化到 8-bit 后的唯一色数 | `< 100` 且 `textured_pixel_ratio` 低时告警 |

第三层（目视）与第一/二层的阈值共同作为数据门：`validate_components.py` 在 `scene["passes"]` 存在时
做结构性检查（始终报错）与质量检查（`require_textures=true` 报错，否则写入 `counts["pass_warnings"]`）。

#### 第三层：抽样目视（人工）

挑 3～5 个**已知带贴图**的物体，把 `albedo.exr` 的线性值按 sRGB（gamma 2.2）转成 PNG，与源
`*_BaseColor.dds` 并排比对。重点看两件事：

1. **接没接上**：标签/文字是否可辨；纯色/品红 = 没接上。
2. **接没接错**：颜色空间必须是 **sRGB**。若被误当 Non-Color，图像会整体偏亮/偏暗但**仍有纹理**，
   这一层只能靠目视发现（第一/二层的统计无法区分）。

### 10.5 sRGB 陷阱

`albedo` 通道必须保持**线性**（EXR 里存的就是线性值）。若要导出 PNG 做目视，务必显式做
`sRGB = linear^(1/2.2)`（或标准 sRGB 传输函数）后写入；把线性值直接当 sRGB 写、或把 sRGB 图片当
线性读，都会得到偏亮/偏暗的结果。DDS 基础色贴图通常是 sRGB 编码，Blender 里对应 Image Texture 节点
应设为 `sRGB` colorspace；若脚本误设为 `Non-Color`，渲染出的 albedo 会偏亮，且第二层统计仍判定“有纹理”。

### 10.6 开关与成本

```yaml
m4:
  render:
    passes:
      enabled: false              # 默认关闭，现有数据逐字节不变
      albedo: true
      normal: true
      irradiance: true
      irradiance_mode: derived    # derived | override
      normal_space: camera        # camera | world
      save_normal_world: false
      pass_samples: 1
      eps: 0.001
      denoise_passes: false
      require_textures: false
      min_base_color_linked_ratio: 0.9
      max_magenta_ratio: 0.0001
```

成本（`derived` 模式）：`+2` 次 1-sample 渲染/job（albedo、normal 参考），`+~19` 个 EXR/job。
512² RGB half ≈ 1.5 MB/张 → 约 +28 MB/job；960² 约 +100 MB/job；时间增量预计 < 10%。
`override` 模式约 ×2 渲染时间，仅作对照。回滚只需把 `passes.enabled` 改回 `false`，代码路径与现状一致。

### 10.7 验收命令

```bash
# P0：纯 NumPy 单测（无需 GPU / Blender）
python -m unittest discover -s datacompose/tests -p 'test_intrinsic_passes.py' -v

# 渲染 1 个 job 的 passes（服务器，需 Blender 4.5）：
python -m datacompose.cli render --blender-bin /opt/blender-4.5/blender
# 然后在 config 里把 m4.render.passes.enabled 置 true 重跑该 job，
# 人工比对 3 个带贴图物体的 albedo 与 *_BaseColor.dds。

# P2：数据门
python -m datacompose.tokenlight.validate_components --config <dataset-config>.yaml
```

验收标准（P2 结束）：10 个 job 全部通过 `validate_components`；`albedo × irradiance ≈ component − dark`
在 `albedo>0.1` 像素上的相对误差中位数 < 5%；`magenta_ratio = 0`；单 job 输出增量在预算内；
关闭开关后产物与当前一致。

