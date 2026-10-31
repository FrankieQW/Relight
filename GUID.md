# datacompose 数据合成指南

本目录是 LightConstruction 的**数据合成**代码集合：把 Objaverse 单物体资产放进或替换到 Bistro 场景里，
渲染出 TokenLight 需要的线性光照分量与 manifest。坐标约定与仓库根目录 `transfer.py` 完全一致，并在此
基础上补齐了**相机坐标系下的图像平面坐标与深度**，因此下游既可以读三维相机向量，也可以读二维图像平面
位置加深度的组合。

- 参考实现：根目录 `transfer.py`（保留原样，作为坐标约定的权威脚本）。
- 模块化实现：`datacompose/coordinates.py`（本包内，供流水线和测试直接导入，并多写 `image_plane`）。
- 一致性保证：`tests/test_coordinates.py` 用同一份 manifest 同时跑两套实现，断言 `camera_position`
  逐位相等；GUID 中的公式与这两个文件同源。

---

## 1. 目录结构

```text
datacompose/
├── GUID.md                     本文档
├── __init__.py                 __version__
├── __main__.py                 python -m datacompose
├── cli.py                      M1–M4 统一入口
├── config.py                   YAML 配置与项目根解析
├── coordinates.py              相机坐标 / 图像平面 / 深度（transfer.py 的模块版 + 扩展）
├── io_utils.py                 原子写、digest、JSONL
├── schemas.py                  Pydantic 数据协议
├── scene_semantics.py          类别归一化与别名
├── object_index.py             prepare-objects
├── scene_extract.py            prepare-scenes（调用 blender/entry.py）
├── llm_annotate.py             annotate-construction（vLLM）
├── geometry_prepare.py         prepare-geometry
├── render_jobs.py              build-render-jobs
├── render_compositions.py      render（调度 blender/render_composition.py）
├── render_recovery.py          渲染 partial 的清理与恢复
├── blender/
│   ├── entry.py                FBX 导入、归一化、稳定 ID、scene fragment
│   └── render_composition.py   组合放置/替换、相机与灯光选择、分量渲染、metadata
├── tokenlight/
│   ├── render_assets.py        Blender 侧渲染工具（供 render_composition.py 动态加载）
│   ├── build_manifests.py      components → train/validation/test.jsonl（含 split 隔离）
│   └── validate_components.py  分量、组合契约与坐标约定校验
└── stage2/
    ├── config.py               stage2 配置物化、冻结标注校验、smoke 配置生成
    ├── run_stage2.sh           M1–M5 编排（data / smoke / train / eval / tests）
    └── stage2.env.example      运维环境变量模板
```

`Lumina-T2X/lumina_next_t2i/`、`Lumina-T2X/tools/tokenlight_data/inspect_dataset.py`、`flux/` 与
`new_method/` 属于**训练与消费侧**，不在本目录内。`inspect_dataset.py` 需要导入
`Lumina-T2X/lumina_next_t2i/tokenlight`，因此仍留在原处，由 `run_stage2.sh` 以该目录为工作目录调用。

---

## 2. 数据流

```text
data/object/object.json ─┐
data/scene/scene.json ───┼─ annotate-construction ─→ annotation_construction.json
                         │                                  │
                         ├── prepare-geometry ─→ prepared_geometry.json
                         │                                  │
                         └── build-render-jobs ─→ render_jobs.jsonl
                                                        │
                                     render（Blender 组合 + 渲染）
                                                        │
              components/<job_id>/{ambient,dark,point_lights,diffuse,in_scene_lights}.exr
              components/<job_id>/metadata.json      ← canonical + world 坐标在这里落盘
                                                        │
                          build_manifests → validate_components → inspect_dataset
                                                        │
              manifests/{train,validation,test}.jsonl  ← 写入 camera_position / image_plane
```

一个 render job 是 `(scene, object, place|replace, target_entity, seed)`。每个 job 都从只读 base
`.blend` 重新打开，渲染完回滚，不把上一个样本的状态带进下一个。

渲染产物里的 `position` 是 canonical 坐标系；训练需要的 OpenCV 相机坐标由 `build_manifests.py` 在
生成本文件时写入 `camera_position`，因此 **manifest 才是训练该读的那份数据**，`components/*/metadata.json`
保持为原始渲染记录。

---

## 3. 坐标系

渲染链路上同时存在四个坐标系。理解它们之间的换算，是读懂 manifest 里灯光坐标的前提。

| 名称 | 轴向 | 原点 | 单位 | 谁写入 |
|---|---|---|---|---|
| Blender world | 右手，`+Z` 向上 | 场景世界原点 | 米 | Blender |
| canonical | `x=right, y=camera-forward, z=up` | `camera.target` | 米 | `render_composition.py` |
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

`render_composition.py` 用 `_world_to_canonical()` 把灯光世界坐标写进 `position`，随后 `transfer.py`
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

`validate_components.py` 会检查 `canonical.position_axes` 是否等于约定值；在
`require_composition_contract: true`（`run_stage2.sh` 的正式数据路径固定开启）下，还会要求每个
`point_lights` 与 `in_scene_lights` 都带 `camera_position`，并校验它与 `position` 的换算结果一致，
缺失或不一致的 manifest 会被数据门直接拒绝。

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

```bash
# 统一入口（cwd 为仓库根目录，或先 pip install -e .）
python -m datacompose.cli prepare-objects      --config configs/default.yaml
python -m datacompose.cli prepare-scenes       --config configs/default.yaml --blender-bin /opt/blender-4.5/blender
python -m datacompose.cli annotate-construction --config configs/default.yaml --base-url http://127.0.0.1:8000/v1
python -m datacompose.cli prepare-geometry     --config configs/default.yaml
python -m datacompose.cli build-render-jobs    --config configs/default.yaml
python -m datacompose.cli render               --config configs/default.yaml --blender-bin /opt/blender-4.5/blender
python -m datacompose.cli clear-render-partial --config configs/default.yaml --job-id <job_id>
```

Blender 侧的调用路径由 `scene_extract.py` 和 `render_compositions.py` 以
`<config.root>/datacompose/blender/*.py` 拼出；`render_composition.py` 再从自身文件位置解析
`tokenlight/render_assets.py` 与 `render_recovery.py`，因此可以从任意工作目录启动 Blender。

完整编排：

```bash
cp datacompose/stage2/stage2.env.example datacompose/stage2/stage2.env
vim datacompose/stage2/stage2.env
STAGE2_ENV_FILE=datacompose/stage2/stage2.env bash datacompose/stage2/run_stage2.sh data
STAGE2_ENV_FILE=datacompose/stage2/stage2.env bash datacompose/stage2/run_stage2.sh tests
```

`run_stage2.sh` 把 `PYTHONPATH` 设为仓库根目录，于是 `python -m datacompose.cli` 与
`python -c "import datacompose"` 都可用。

---

## 7. 验收

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

坐标相关的检查集中在 `tests/test_coordinates.py`：

- 同一 manifest 上，`datacompose.coordinates.convert_row()` 与根目录 `transfer.py` 的
  `convert_row()` 产出逐位相等的 `camera_position`；
- 两者对 `renderer_position` 不一致的 manifest 都报错；
- `canonical ↔ camera`、`world ↔ camera` 往返一致；
- `depth` 与欧氏距离、与“到成像平面距离”的差异被显式断言；
- `u/v` 与 `fx/fy/cx/cy` 的投影公式一致，`image_plane_to_camera()` 能无损还原相机坐标；
- 镜头平移只改变主点，不改变焦距；
- 目录转换是原子的，且默认拒绝覆盖。

Blender 侧另有 `tests/blender_composition_focus.py`，用真实相机验证取景、遮挡与灯具可见性：

```bash
/opt/blender-4.5/blender --background --factory-startup --python tests/blender_composition_focus.py
```

---

## 8. 与旧路径的映射

| 旧位置 | 新位置 |
|---|---|
| `src/lightconstruction/*.py` | `datacompose/*.py` |
| `src/lightconstruction/cli.py` | `datacompose/cli.py` |
| `scripts/blender_entry.py` | `datacompose/blender/entry.py` |
| `scripts/blender_render_composition.py` | `datacompose/blender/render_composition.py` |
| `scripts/stage2_config.py` | `datacompose/stage2/config.py` |
| `scripts/run_stage2.sh` | `datacompose/stage2/run_stage2.sh` |
| `scripts/stage2.env.example` | `datacompose/stage2/stage2.env.example` |
| `Lumina-T2X/tools/tokenlight_data/render_assets.py` | `datacompose/tokenlight/render_assets.py` |
| `Lumina-T2X/tools/tokenlight_data/build_manifests.py` | `datacompose/tokenlight/build_manifests.py` |
| `Lumina-T2X/tools/tokenlight_data/validate_components.py` | `datacompose/tokenlight/validate_components.py` |
| `python -m lightconstruction.cli` | `python -m datacompose.cli` |
| `from lightconstruction.x import y` | `from datacompose.x import y` |
| `from scripts.stage2_config import y` | `from datacompose.stage2.config import y` |

`transfer.py` 仍保留在仓库根目录，作为坐标约定的参考实现；`datacompose/coordinates.py` 是它的模块化
版本，多写 `image_plane`，其余数值行为与它一致。

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
