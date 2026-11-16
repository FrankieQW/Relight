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
├── prepare_scenes.py           python prepare_scenes.py         → prepare-scenes
├── render_dataset.py           python render_dataset.py         → render
├── clear_render_partial.py     python clear_render_partial.py   → clear-render-partial
├── plan_scene_cameras.py       python plan_scene_cameras.py     → plan-scene-cameras
├── probe_scenes.py             python probe_scenes.py           → probe-scenes（需 Blender）
├── probes.py                   三角面距离场探针：烘焙数据格式 + 纯 NumPy 查询
├── config.py                   YAML 配置与项目根解析
├── coordinates.py              相机坐标 / 图像平面 / 深度（transfer.py 的模块版 + 扩展）
├── intrinsic.py                物理表征通道推导（albedo / irradiance / normal，纯 NumPy）
├── io_utils.py                 原子写、digest、JSONL
├── schemas.py                  Pydantic 数据协议（场景 + 相机 job）
├── scene_semantics.py          类别归一化与别名
├── scene_cameras.py            场景内无碰撞相机采样（AABB 宽相 + 空心容器 + 视野无穿插判据）
├── scene_targets.py            目标实体挑选 + 投影/可见性几何（实体条件化取景）
├── scene_extract.py            prepare-scenes（调用 blender/entry.py）
├── scene_probe.py              probe-scenes（调度 blender/probe_scene.py，逐场景烘焙探针）
├── render_scenes.py            render（多 GPU worker 调度 blender/render_scene.py）
├── render_recovery.py          渲染 partial 的清理与恢复
├── transfer.py                 坐标约定权威参考实现（纯标准库）
├── configs/
│   ├── default.yaml            项目配置（project.root=.. 指向本目录）
│   ├── category_aliases.yaml
│   └── scene_overrides.yaml
├── blender/
│   ├── entry.py                FBX 导入、归一化、稳定 ID、scene fragment
│   ├── probe_scene.py          场景烘焙成三角面 clearance 探针（npz）
│   └── render_scene.py         相机位姿精确复核、光照、分量渲染、metadata
├── tokenlight/
│   ├── render_assets.py        Blender 侧渲染工具（供 render_scene.py 动态加载）
│   ├── build_manifests.py      components → train/validation/test.jsonl（按 scene 切分）
│   └── validate_components.py  分量完整性与坐标约定校验
├── scripts/
│   └── remove_legacy_object_pipeline.sh   清理 object 时代残留文件
├── tools/
│   ├── bench_scene_cameras.py  相机搜索成本基准
│   └── probe_smoke_test.py     探针烘焙冒烟测试（需 Blender，无数据）
├── docs/
│   └── scene_only_render_feasibility.md   纯场景合成可行性方案
└── tests/
    ├── test_coordinates.py
    ├── test_intrinsic_passes.py
    ├── test_manifest_camera.py
    ├── test_render_recovery.py
    ├── test_launcher.py
    ├── test_scene_cameras.py
    ├── test_scene_camera_targets.py
    ├── test_scene_containers.py
    └── test_scene_probe.py
```

本目录是**独立可复制的 compose 项目**：只依赖自身代码与本目录下的 `configs/`、`transfer.py`，
不引用上级仓库的 `src/`、`scripts/`、`Lumina-T2X/`。场景数据（Bistro 等 FBX）与训练侧
由外部提供，通过 `--config` 的数据路径接入；原 `stage2`（训练编排）已移出本项目。

---

## 2. 数据流

```text
data/scene/scene.json ── plan-scene-cameras ─→ scene_camera_jobs.jsonl
        │                                             │
        └─ probe-scenes（可选）                        render（Blender：位姿复核 + 渲染）
           → <scene_id>.probe.npz  ──┐                       │
                                     └───────────────────────┤
              components/<job_id>/{ambient,dark,point_lights,diffuse,in_scene_lights}.exr
              components/<job_id>/metadata.json      ← canonical + world 坐标在这里落盘
                                                      │
                          build_manifests → validate_components
                                                      │
              manifests/{train,validation,test}.jsonl  ← 写入 camera_position / image_plane
```

一个 render job 是 `(scene, camera pose, lighting profile, seed)`，与物体无关。每个 job 都从只读 base
`.blend` 重新打开，渲染完回滚，不把上一个样本的状态带进下一个。

相机位姿由 `scene_cameras.py` 规划，再由 `blender/render_scene.py` 用真实三角面复核：
穿过几何体、贴脸、只对着一面墙、深度过平都会让这个 job 直接失败，而不是产出一张坏图。

规划侧有两种几何判据，由 `m4.camera.free_space.source` 选：

- `aabb`：只用每个节点的世界 AABB（宽相）。便宜、保守，但**盒子永远高估物体**，所以 L 形房间会被
  当成矩形、多层建筑的相机可能落到隔壁楼层、一个“被误判成空心”的箱子会让相机站进墙里。
- `probe`：用 `probe-scenes` 烘焙的三角面距离场（见 7.2）。位置从“已验证自由的体素列”里采样，
  视线也用同一份场来投射，因此“规划器认为自由”和“渲染器认为自由”是同一个判据。

`auto`（默认）有探针就用、没有就退回 AABB，所以**没烘焙探针的数据集行为不变**——这一点由
`free_space.source: aabb` 再加一层显式保证。

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

#### 4.1.1 相机策略与目标实体

`camera.strategy` 决定这些相机是怎么选出来的：

| 值 | 含义 |
| --- | --- |
| `scene_free_space_v1` | 原始策略：在场景自由空间里随机取位置+朝向。默认值，旧数据集可用它逐字节复现。 |
| `entity_conditioned_v1` | 以命名实体为目标绕球壳取景，并混入 `mix_with_free_space` 比例的远景。 |

**空心容器模型（`m4.camera.container.model`）**：进口的室内场景常常是“一整块网格”（单个节点，
AABB 覆盖整个房间）。在“盒子=实心”的自由空间模型里，这种场景的每一个眼高点都在几何体内部，
从室内发出的射线也永远打不到壳体——于是整个房间规划不出任何相机（真实 Bistro 片段实测 4400 次
尝试 0 通过）。`container_v1` 把这种盒子标记为**空心**，只改变三件事：

1. `clearance`：点在空心盒内时取“到最近一个面的距离”（即到墙的距离），而不是 0；
2. `raycast`：从空心盒内射出的射线命中的是**对面墙**（出口面），而不是穿过；
3. dominance：**只当相机位于该容器内**时，容器的命中不计入 `single_node_fraction`
   （房间是环境，不是“一个占满画面的物体”）。相机在容器外时行为与旧模型完全一致，室外场景不受影响。

容器判据是纯几何的（不看 category，因为 category 只是分类器输出）：三轴都 ≥ `min_extent`、
体积 ≥ `min_volume`、AABB 内至少含 `min_contained_nodes` 个其它节点中心。把
`container.model` 设成 `aabb_legacy` 即恢复旧的实心盒模型，用于逐字节复现旧数据集。

> ⚠️ 容器模型**默认开启**，对两种策略都生效（`scene_free_space_v1` 在室内场景上一模一样地坏过）。
> 因此想复现旧数据集要**同时**确认 `camera.container.model: aabb_legacy` 与
> `camera.strategy: scene_free_space_v1`；另外修复轴平行射线伪命中（见下）会让结果与旧提交
> 不是严格逐字节一致，语义上是修正而非回归。

配套的两处修正：

- **`camera_outside_room`**：水平落在某个容器 footprint 内、但垂直超出它的 z 范围时拒绝。
  这正是“相机飘到屋顶以上看天空”的来源；水平在容器外则放行，保护混合场景的室外部分。
- **地板来源优先级**：`support_categories` 实体 → 容器 `min_z`（导入室内的地板本身就在壳体网格内，
  所以室内往往**没有** floor 实体）→ `free_height_band` 配合绝对值钳制 `free_height_clamp`。

目标挑选是**几何判据 + 打分排序**，category 只做早期剔除和排序惩罚，不做白名单：硬过滤
`min_thickness` / `flat_ratio` / `max_dimension` / `max_scene_diagonal_fraction`，
排序用 `framability_score`。注意 `min_target_screen_ratio` **不再用于过滤目标**：绕行半径
`r = radius_scale × 半对角线` 会让尺寸约掉，估计值对所有候选都恒为 ~1.0。

`entity_conditioned_v1` 的 job 会额外写：

```json
"camera": {
  "strategy": "entity_conditioned_v1",
  "target_entity_id": "scene_a:entity:0f3c...",
  "target_label": "CoffeeCup_01",
  "target_visible_ratio": 0.83,
  "target_screen_ratio": 0.14
},
"canonical": {
  "origin": [12.4, -3.1, 0.82],
  "asset_size": 0.62,
  "target_entity_id": "scene_a:entity:0f3c...",
  "target_label": "CoffeeCup_01",
  "target_radius": 0.104,
  "target_node_ids": ["node_00412"]
},
"free_space_report": {
  "occluder_node_ids": ["node_00007"],
  "room_node_id": "node_00001",
  "container_node_ids": ["node_00001"]
},
"lineage": {"target": {"entity_id": "...", "label": "CoffeeCup_01", "visible_ratio": 0.83, "...": "..."}}
```

要点：

- `canonical.origin == camera.target` 仍然成立，但此时它就是**目标实体的中心**，不再是几何中位深度
  上的空气点。
- `canonical.asset_size` **保持原语义**：相机到 anchor 的距离。目标尺寸放在 `canonical.target_radius`，
  因此渲染侧 `anchor_depth` 与点光/面光的距离缩放不需要任何改动。
- `target_visible_ratio` 是规划阶段的 AABB 可见性估计；渲染侧用真实三角形复核后会写
  `metadata.target.geometry_visible_ratio`，两者不一致时以渲染侧为准。
- 没有 `target_entity_id` 的 job 表示它是自由空间远景（混采），构图门槛是“画面里至少有一个目标
  真正可见”，因此不会是纯墙角。
- `free_space_report.container_node_ids` 会被 `blender/render_scene.py` 用来把容器对象**从 dominance
  里排除**（仍参与 near plane / depth spread），避免规划器放行、渲染侧 `single_object_dominates` 误拒。

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
python render_dataset.py       --clear-failed        # 重试前清掉本批失败 partial
python clear_render_partial.py --job-id <job_id>     # 只清单个 job
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
`tests/test_scene_camera_targets.py` 覆盖实体条件化取景：纯墙角场景必须产不出相机、目标可见性与
画面占比门槛、被完全遮挡的目标被拒、接触容差、30% 混采的分配与 job/schema 契约。
`tests/test_scene_containers.py` 覆盖空心容器模型：容器检测、室内 clearance/raycast、`outside_room`、
`aabb_legacy` 复现旧的“室内全灭”、三种地板来源的优先级、目标几何过滤（薄片/超大/跨度）、
`framability_score` 排序、以及轴平行射线伪命中的回归；同时校验 `configs/default.yaml` 里
`m4.camera` / `camera.target` / `camera.container` / `camera.free_space` / `m4.probe` 的键
与 dataclass 字段一致（防止配置与代码漂移），并断言这些新阈值的默认值不改变旧行为。
`tests/test_scene_probe.py` 覆盖探针：距离场数值（用解析 AABB 距离做真值）、`npz` 往返、查询是
保守下界、`cast` 命中/逃逸与节点归属、`support_z` 推地板、多层建筑两层都能出机位、
`ProbeFreeSpace` 与 AABB 的节点/容器契约、软判据 `view_coverage`、`min_coverage` 拒绝、
以及规划侧四种状态（`probe` / `no_probe` / `source_aabb` / `probe_stale`）；最后用一个
假的 Blender 替身把 `probe-scenes` 端到端跑通（产出、`probe_summary.json`、resume 跳过、
单场景失败只发警告、`require_all` 才抛错）。

### 7.1 相机策略调参

`m4.camera.strategy` 切成 `entity_conditioned_v1` 后，所有阈值都在 `m4.camera.target` 下，改完重跑
`plan_scene_cameras.py` 即可，不需要改代码。最常动的几个：

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `target.min_target_visible_ratio` | 0.40 | 目标表面有多少比例必须真正看得见。贴桌/贴墙的物体不要调高，调高会只剩悬空物体。 |
| `target.radius_scale` + `radius_jitter` | 2.6 / `[0.85, 1.50]` | 绕行半径 = `radius_scale × 半对角线`。数值越大越"远景"，越小越"特写"。 |
| `target.min_radius_fraction` | 0.70 | 靠墙/贴桌的目标不要被“按最近障碍物”的半径上限压到特写，这是该上限的下限比例。 |
| `target.min_azimuth_separation_degrees` | 15.0 | 同一目标两个机位的最小角度间隔。**绕行取景不能用欧氏间距**：12 cm 的杯子在 0.6 m 轨道上，任意两个机位最多相距 1.2 m，1.5 m 的 `min_camera_spacing` 会让每个目标只能出 1 个机位。 |
| `mix_with_free_space` | 0.30 | 远景占比。设为 0 就全部是目标特写。 |
| `target.max_dimension` | 4.00 | 最长轴超过这个（米）就不算可构图目标：挡掉街道、地形、建筑壳体。 |
| `target.flat_ratio` / `min_thickness` | 0.06 / 0.03 | 扁平度与厚度下限：挡掉门垫、路面这类“片”。 |
| `camera.container.model` | `container_v1` | 设成 `aabb_legacy` 恢复旧的实心盒模型；旧数据集复现用它。 |
| `camera.container.clearance_margin` | 0.10 | 室内额外要求的墙距。规划器用矩形近似房间，可能高估墙距，渲染侧因此误拒时可调大。 |
| `camera.free_height_clamp` | `[1.00, 2.50]` | 无地板、无容器时的高度绝对值钳制，防止相机飞到屋顶以上。 |

调参前先用离线基准跑一遍 reject 分布（无需 Blender）：

```bash
python -m datacompose.tools.bench_scene_cameras --strategy entity_conditioned_v1 --cameras 8
```

`reject_reasons` 里 `target_too_large` 偏多说明半径偏小，`target_occluded` 偏多说明
`min_target_visible_ratio` 太高或遮挡体判得太死，`no_geometry_in_view` 偏多说明
`min_hit_fraction_targeted` 太高，`camera_inside_geometry` 偏多而场景明显是室内时检查
`camera.container.model` / `min_contained_nodes`（容器没被识别出来）。

### 7.2 探针阶段（`probe-scenes`）

`AABB` 是宽相：盒子永远包含自己的网格，所以它分不清“空房间的壳”和“实心箱子”，
也表达不了 L 形房间。这类近似每一个都会变成“规划器接受、渲染器拒绝”的废样本。
彻底的做法是**直接问几何**，只问一次，然后把答案缓存下来——这也是 BlenderProc 的
`probe` / `check_novel_pose` 一类做法的思路。

```bash
python probe_scenes.py --blender-bin /opt/blender-4.5/blender --workers 4
```

一个 `(scene)` 的产物是布局在 base `.blend` 旁边的 `<stem>.probe.npz`：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `format` | str | `lc_probe_v1`，读的时候不匹配直接拒绝 |
| `origin` / `resolution` | float64 | 体素网格原点与边长（米） |
| `clearance` | float32 `(nx,ny,nz)` | 到最近三角面的距离，钳到 `m4.probe.clearance_cap` |
| `nearest_node` | int32 | 该最近三角面属于哪个 `scene.json` 节点（`node_ids` 下标） |
| `node_ids` | str[] | 烘焙时写下的节点 id |

三个关键设计点：

- **不需要内外判定。** “到最近三角面 ≥ `camera_radius`”本身就是判据——距离场对墙体两侧是对称的，
  所以一个点只要离所有三角面都够远，它就不可能在几何体里。唯一漏掉的情形是“实心墙比两倍相机半径
  还厚”，而 `render_scene.py` 在渲染前仍会逐位复核，漏网的只会浪费一个 job，不会产出坏图。
- **地板是从距离场里推出来的，不额外烘焙。** 一列体素里 clearance 第一次超过阈值的位置就是“可以站的
  地面”，而这个体素属于哪一段连续自由段，就决定它属于哪一层楼。多层建筑因此自然正确。
  这也替代了旧的 `_floor_boxes` / `outside_room` 两条启发式。
- **查询是保守下界。** 距离场是 1-Lipschitz 的，所以“取最近体素的值、再减去查询点到该体素中心的距离”
  一定不超过真值。规划器永远不会被告知“这里比实际上更宽敞”。

  代价是**系统性低估**：实测（10×10×3 房间，13 cm 体素，3000 个随机点）平均低估 6 cm、
  最坏 17 cm（约 1.3 × 体素边长），**高估 0 次**——所以只会误拒、不会误收。
  想减少误拒就调小 `m4.probe.resolution`（成本线性上升），或者用 AABB 粗筛后只对候选做精细判定。

`container`（谁是房间壳）仍然由 AABB 提供：dominance 判据必须把房间本身排除掉，否则所有室内视图
都会因为“所有射线都打在墙上”而失败。这个猜测出错现在只是诊断数字不准，`ProbeFreeSpace` 只把
`container_at` / `outside_room` 委派给 AABB，碰撞与射线全部走探针。

调参表：

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `m4.probe.resolution` | 0.10 | 体素边长（米）。10 cm 能分辨椅子腿和门框；超大碎片调到 0.2 省一半时间。 |
| `m4.probe.max_voxels` | 1_500_000 | 体素总数上限。装不下就自动加粗边长（×1.15），**烘焙成本由它决定**（每体素约 10–20 µs）。 |
| `m4.probe.clearance_cap` | 1.00 | 距离截断。规划器不需要知道某点离墙 20 m，只需要知道它不贴着墙。 |
| `m4.probe.padding` | 0.10 | 包围盒外扩容差（米）。必须 > 0，否则外壳三角面落在网格边界上。 |
| `m4.probe.resume` | true | 探针比 `.blend` 新就跳过。 |
| `m4.probe.require_all` | false | `false` 时单场景烘焙失败只记进 `probe_summary.json`，该场景退回 AABB。 |
| `m4.camera.free_space.source` | `auto` | `auto`/`probe`/`aabb`。旧数据集复现用 `aabb`。 |
| `m4.camera.free_space.hit_clearance` | 0.10 | 射线“算被挡住”的距离阈值。必须 ≥ 行进步长，否则会穿透。 |
| `m4.camera.free_space.ray_step_scale` | 0.5 | 行进步长 = 体素边长 × 该系数。 |
| `m4.camera.free_space.min_node_coverage` | 0.90 | 探针能识别的节点比例下限。低于它就认为探针过期（`probe_stale`），退回 AABB——用别的 `.blend` 烘出来的探针会把每条射线都归错节点。 |

规划摘要里能看到每个场景到底用了哪条路：`stats.free_space.{source,probe_scenes,fallback_scenes,status}`，
每个 job 的 `free_space_report.source` 也会写下 `probe` 或 `aabb`。

成本参考（`--probe` 开关会在离线基准里打出这两行，测的是**查询**成本，不是烘焙）：

```bash
python -m datacompose.tools.bench_scene_cameras --probe --cameras 8
# pose_evaluation=196.21 ms         ← AABB：遍历 4625 个盒子
# probe_pose_evaluation=37.22 ms    ← 探针：查表
```

探针查询比 AABB 路径**快约 5 倍**（同样的 576 条射线），因为盒子路径每条射线都要跟每个节点求交，
而探针只查最近体素。烘焙是另一个数量级的开销：`find_nearest` 是逐体素的 Python 循环，
36 个三角面的玩具场景实测 **1.4 µs/体素**（118 万体素 1.7 s），真实碎片三角面上百万时
每体素会慢几倍；`max_voxels` 默认 150 万，所以一个场景是**秒到一分钟**量级。

烘焙这条路径没法用单元测试覆盖（要真的 `BVHTree` 和 `.blend`），所以在目标 Blender 上先跑一次冒烟：

```bash
python tools/probe_smoke_test.py --blender-bin /opt/blender-4.5/blender
```

它会现场搭一个 10×10×3 的空心房间 + 桌子 + 杯子，烘焙，再拿解析真值校验 clearance /
射线命中节点 / 地板高度，最后打印 `PASS`。换 Blender 版本、换机器时先跑这个。

### 7.3 软判据（默认关闭）

两处借鉴 BlenderProc 的“软打分”，都**只记录、不参与判定**，除非显式给权重：

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `m4.camera.coverage_weight` | 0.0 | `coverage`（类别多样性 − 单物占比）加进位姿评分的系数。 |
| `m4.camera.min_coverage` | 0.0 | `coverage` 硬门槛。非 0 时新增一条 `low_scene_coverage` 拒绝。 |
| `m4.camera.target.azimuth_novelty_weight` | 0.0 | 同一目标的机位按“与已选方位角的角距”加权排序。**不是**硬门槛。 |

三者默认 0，所以接受/拒绝结果与旧数据集一致；每个位姿的 `coverage` / `azimuth_novelty`
始终会算出来并写进 metrics，可以直接看分布再决定要不要打开。

关于许可：本节与 `probes.py` / `blender/probe_scene.py` 只借鉴 BlenderProc 的**思路**
（用真实几何校验候选、用覆盖度打分、用位姿新颖度排序），没有复制其代码——BlenderProc 是
GPL-3.0，抄进来会把整条管线一起传染。

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
  只有 `albedo` 与 `normal` 同时开启时才会产出。

> **当前实现范围**：`irradiance_mode` 只支持 `derived`，`normal_space` 只支持 `camera`；
> 设为 `override` / `world` 会在渲染开始时直接报错，而不是静默回退。
> 用白色朗伯材质覆盖渲染的 `override` 方案尚未实现，渲染时间不会 ×2。

### 10.2 产出布局

```text
components/<job_id>/
└── passes/
    ├── albedo.exr              # albedo:true 时产出（Diffuse Color）
    ├── normal_camera.exr       # normal:true 时产出（相机空间单位法线）
    ├── normal_world.exr        # 可选（normal:true 且 save_normal_world:true）
    ├── valid_mask.png          # albedo 与 normal 同时开启时产出，8-bit
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
  "enabled": {"albedo": true, "normal": true, "irradiance": true, "normal_world": false},
  "albedo": "components/<id>/passes/albedo.exr",
  "normal_camera": "components/<id>/passes/normal_camera.exr",
  "valid_mask": "components/<id>/passes/valid_mask.png",
  "material_audit": "components/<id>/passes/material_audit.json",
  "quality": {"base_color_linked_ratio": 0.98, "magenta_ratio": 0.0, "textured_pixel_ratio": 0.34},
  "irradiance": {"ambient": "...", "point_000": "...", "fixture_000": "..."}
}
```

`enabled` 记录本 job 实际产出了哪些 pass；`albedo` / `normal_camera` / `valid_mask` /
`normal_world` 这些键只在对应开关打开时出现，`validate_components.py` 也按 `enabled` 检查，
因此关闭单项开关不会产生“metadata 指向不存在文件”的假错误。

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
      albedo: true                # albedo 与 normal 至少开一个
      normal: true
      irradiance: true            # 需要 albedo:true
      irradiance_mode: derived    # 目前仅实现 derived
      normal_space: camera        # 目前仅实现 camera
      save_normal_world: false    # 需要 normal:true
      pass_samples: 1
      eps: 0.001
      denoise_passes: false       # 预留，当前未生效
      require_textures: false
      min_base_color_linked_ratio: 0.9
      max_magenta_ratio: null     # null = 不做品红阈值检查
```

成本（`derived` 模式）：`+2` 次 1-sample 渲染/job（albedo、normal 参考），`+~19` 个 EXR/job。
512² RGB half ≈ 1.5 MB/张 → 约 +28 MB/job；960² 约 +100 MB/job；时间增量预计 < 10%。
回滚只需把 `passes.enabled` 改回 `false`，代码路径与现状一致。

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

# P2（可选，需要 opencv-python）：同时校验像素与 albedo×irradiance 恒等式
python -m datacompose.tokenlight.validate_components --config <dataset-config>.yaml --pixel-checks
```

验收标准（P2 结束）：10 个 job 全部通过 `validate_components`；`albedo × irradiance ≈ component − dark`
在 `albedo>0.1` 像素上的相对误差中位数 < 5%；`magenta_ratio = 0`；单 job 输出增量在预算内；
关闭开关后产物与当前一致。

> 注：`validate_components.py` 默认做结构检查（文件存在、material_audit、按 `enabled` 的字段齐全、
> 每个 point/fixture 分量都有对应 `irradiance`）与质量阈值检查。加上 `--pixel-checks` 后会解码
> albedo / normal / irradiance 像素，校验法线单位长度、irradiance 非负，并逐分量验证
> `albedo × irradiance_L ≈ component_L − dark`（`albedo>0.1` 且 `valid` 的像素，中位相对误差 < 5%）。
> 像素检查需要 `opencv-python`；未安装时默认模式仍可运行。
>
> 注意：`--pixel-checks` 目前只接受 `.npy` 或 OpenCV 能解码的 EXR；如果 OpenCV 未编译 OpenEXR 支持，
> 会直接报解码错误而不是静默跳过。

