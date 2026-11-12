# 纯场景合成（Scene-only Synthesis）可行性方案

> 目标变更：放弃 *Objaverse 物体 + LLM 语义配对 + 摆放/替换* 的数据合成路线，
> 改为**直接把已有场景（Bistro）在不同光照条件下渲染**。组合空间从
> `object × scene × target × relation × camera` 收缩为 `scene × camera × lighting`，
> 唯一需要求解的工程问题变成**相机位姿**：相机必须站在自由空间里、且视野内不与任何
> 几何体发生穿插（穿墙/穿桌/贴脸）。

本文档基于对本仓库现有代码的实测（见第 4 节），不是纸面推演。

---

## 0. 结论

**可行，且整体复杂度低于现方案。** 理由三条：

1. **地基已经存在。** 新方案需要的场景侧资产（归一化 `.blend`、`scene.json` 里的
   node/entity/AABB/类别）由 `scene_extract.py` + `blender/entry.py` 产出，一行不用改。
2. **渲染侧 80% 可复用。** `blender/render_composition.py` 里的 Cycles 配置、线性 EXR
   渲染、ambient/dark/逐光源分量、intrinsic passes、metadata、partial→final 原子提交、
   失败恢复，都是与"有没有插入物体"无关的通用能力。
3. **真正的新工作只有一块：相机采样 + 自由空间校验。** 这一块已经写了不依赖 Blender
   的原型 `scene_cameras.py`，并通过 14 项单元测试 + 4625 节点合成场景基准测试
   （搜索成本 **0.19 s / 相机**，相对渲染开销可忽略）。

风险不在"能不能"，而在**样本质量口径**（第 6 节）：没有物体作为显著性锚点后，需要
重新定义"什么样的视角算有信息量"，否则会退化成大量空白墙面的废样本。

---

## 1. 方案定义

一个训练样本 = `(scene_id, camera_pose, lighting_config)`，输出与现状**完全一致**的
TokenLight 分量：

| 分量 | 现状 | 新方案 |
|---|---|---|
| `ambient.exr` | HDRI/常色世界光 | 同（HDRI 维度可扩大） |
| `dark.exr` | 全暗参考 | 同 |
| `point_lights/light_XXX.exr` | 相机视锥内 16 个点光 | 同，范围随深度自适应 |
| `diffuse/spread_XX.exr` | 6 档面光尺寸 | 同 |
| `in_scene_lights/fixture_000_on.exr` | 物体旁的程序化灯具 | **改为场景原生灯具**（更好） |
| `passes/` | albedo / normal_camera / valid_mask / irradiance | 同 |
| `metadata.json` | 组合契约 | 契约字段替换（见 3.7） |

变化的是**采样维度**：不再需要 object 清单、类别几何先验、类别对语义规则、target 实体、
place_on/replace 关系、碰撞收缩缩放。

---

## 2. 与现行方案的代码级差异

| 模块 | 现在做什么 | 新方案处置 |
|---|---|---|
| `object_index.py` | Objaverse 清单∩LVIS 严格交集，80 物体/74 类 | **删除** |
| `geometry_prepare.py` | 类别尺寸先验、up/front 轴、许可证白名单 | **删除** |
| `llm_annotate.py` | LLM 类别对语义判定 + 0.8 置信度门槛 | **删除**（项目里最脆的一环整体退场） |
| `configs/category_dimensions.yaml`、`object_overrides.yaml` | 物体几何/覆盖 | **删除** |
| `scene_extract.py` / `blender/entry.py` | FBX → 归一化 blend + scene.json（nodes/entities/AABB/类别/flags） | **原样保留**，是新方案地基 |
| `scene_semantics.py` | label 归一化、类别兜底 | **保留**（实体类别仍靠它） |
| `render_jobs.py` | object×scene 笛卡尔积 + 关系优先级 | **重写为场景相机 job 生成器**（原型已在 `scene_cameras.py`） |
| `render_compositions.py` | 多 GPU worker 切分、resume、partial 汇总 | **保留骨架**，只换 job 字段 |
| `blender/render_composition.py` | 导入物体、缩放、碰撞收缩、相机选择、渲染、metadata | **拆分改造**：删 `import_asset`/`_normalize_imported`/`_resolve_inserted_placement`/fallback 灯具；**保留**渲染与分量逻辑；`_select_camera` 改为消费已定好的位姿（或直接用 BVH 精校验） |
| `coordinates.py`、`intrinsic.py`、`tokenlight/*`、`transfer.py` | 坐标契约、内禀通道、manifest 构建/校验 | **不改**（原型已按该契约生成 job，见 4.3） |

---

## 3. 核心技术问题与解法

### 3.1 没有物体当锚点，相机看向哪里？

现方案的 `camera.target` 是插入物体的中心，`canonical.origin` 必须等于它（`coordinates.CameraFrame.from_row` 强制校验）。新方案定义：

> **锚点 = 该位姿第一次命中几何体的中位深度点**，即 `target = location + forward * median(hit_depth)`。

这样：

- `canonical.origin == camera.target` 契约天然成立（已在测试中验证）；
- 锚点必然落在**真实可见表面上**，而不是空中的任意点；
- `canonical.asset_size` 复用为"锚点深度"，等价于现状里 subject 的尺度语义，下游不需要改。

如果希望更强的内容约束（例如必须正对某个 `table` 实体、或必须让某类物体占画面 X%），
可以在此之上加"看向某实体"的采样模式——实体信息 scene.json 里已有，不增加标注成本。

### 3.2 "视野内不发生碰撞"的判据（本方案的核心）

需要区分三种失败：**相机在物体内部**、**相机没在里面但贴着几何体**、**画面被单一平面糊满**。
原型实现 6 条 gate（顺序即拒绝原因，全部可统计）：

| Gate | 判据 | 默认阈值 | 解决的失败模式 |
|---|---|---|---|
| `camera_inside_geometry` | 相机点到最近包围盒表面的距离 ≥ `camera_radius` | 0.20 m | 相机埋在桌子/墙里 |
| `no_geometry_in_view` | 视锥射线命中率 ≥ `min_hit_fraction` | 0.90 | 对着天空/空洞，画面无内容 |
| `near_geometry_in_view` | 所有射线的最小命中深度 ≥ `near_clearance` | 0.30 m | **穿模**：有东西比近裁剪面更近 |
| `closeup_single_surface` | 深度 < `near_window` 的射线占比 ≤ `max_near_fraction` | 1.0 m / 0.25 | 贴脸看墙面（虽合法但无信息） |
| `single_node_dominates` | 最近命中里单一 node 占比 ≤ `max_single_node_fraction` | 0.90 | 整幅画面是一块板/天花板 |
| `depth_spread_too_small` | `p90/p10` 深度比 ≥ `min_depth_spread_ratio` | 1.35 | 视角扁平、无空间结构 |

**两层实现，成本互补：**

1. **Python 宽相（本原型）**：用 `scene.json` 已有的 node 世界 AABB 做 slab 射线求交，
   576 条射线 × 4625 box ≈ 90 ms/位姿（12×12 时 ≈ 24 ms）。可在任意机器上跑，不需要 GPU，
   适合"先筛掉 80% 候选"。
2. **Blender 精相（改造时接入）**：复用 `_world_bvh` / `_find_scene_collisions` 的
   `BVHTree.overlap` 与 `ray_cast`，对**已在宽相通过的少数位姿**做三角面级确认，
   并顺手把 `near` 裁剪面（`camera.data.clip_start`）设成 `near_clearance` 的一半。

AABB 宽相是**保守**的（会误杀少量合法位姿，不会放过穿插），因此不会污染数据质量。

### 3.3 站位高度：白捡的真实感

无物体的场景里，最容易被忽略的是"相机漂浮在半空"。`scene.json` 的 entities 已经带有
类别，`floor/sidewalk/road/stair/...` 可直接当站立面：

- 在 XY 上找覆盖该点的地板 box，取其上表面 `z`，再在 `eye_height ∈ [1.10, 1.80] m` 内采样；
- 没有地板覆盖就拒绝（`no_floor_under_candidate`）；
- 若场景完全没有地板类实体，自动退化为 `height_policy: "free"`（按 bbox 高度带采样）。

这条策略零额外标注成本，却直接消灭"相机悬空/贴地/穿地板"三大类废样本。

### 3.4 采样、预算与确定性

- 位置：bbox 内均匀采样（或地板约束采样）→ 位姿：`yaw ∈ [0,360)`、`pitch ∈ [-18,12]°`；
- **两级 ray grid**：搜索用 12×12（144 射线）粗筛，通过者再用 24×24（576 射线）复核，
  复核结果才写进 manifest（实测把搜索成本从 2.76 s 降到 1.55 s，见 4.2）；
- 相机间距约束 `min_camera_spacing`：避免同一场景出现几十个几乎重合的视角；
- 全部随机源由 `(seed, scene_id)` 派生（`sha256` → `random.Random`），**完全可复现**，
  与现方案 `_seed_for` 同一套路。

### 3.5 光照维度（新方案真正新增的"数据多样性"）

| 维度 | 现状 | 新方案建议 |
|---|---|---|
| 环境光 | HDRI 或常色 world | **主力**：`hdri_root` 下按 job 摘要确定性抽 HDRI（现成代码 `_select_ambient_hdri`）；可选 `time-of-day`/色温分组 |
| 场景原生灯 | 只用一个 fixture 代表 | **可枚举**：`fixture_categories`（lamp/lantern/light/chandelier/sconce）在场景里通常是几十个，逐个开/关/组合，天然产生"部分点亮"的多样性 |
| 点光 | 16 个，范围以**物体尺度**为准（x∈[-1.2,1.2] m…） | **必须改成深度自适应**：`z ∈ [0.5·d, 1.2·d]`，`x∈±1.0·d·tan(fov/2)`，`d`=锚点深度，否则光全糊在镜头前 |
| 面光 | 6 档尺寸、固定前上方位置 | 同，位置改为锚点相对 |
| 关闭原生光照 | `_disable_native_lighting()` 全关 | **保留**（分解式监督必须可控），但 scene-render 下要新增"只关 emissive 不关 light object"的开关，方便测试 |

新增配置建议（`configs/default.yaml`）：

```yaml
scene_only:
  cameras_per_scene: 32
  lighting_variants_per_camera: 1
  camera:
    focal_length: 50.0
    yaw_degrees: [0, 360]
    pitch_degrees: [-18, 12]
    eye_height: [1.10, 1.80]
    camera_radius: 0.20
    near_clearance: 0.30
    min_hit_fraction: 0.90
    near_window: 1.00
    max_near_fraction: 0.25
    max_single_node_fraction: 0.90
    min_depth_spread_ratio: 1.35
    ray_grid: 24
    search_ray_grid: 12
    attempts_per_camera: 32
    min_camera_spacing: 1.5
  lighting:
    ambient_source: hdri          # hdri | constant | none
    point_lights_per_scene: 16
    scene_fixtures: true
    fixture_mode: subset          # single | subset | all
```

### 3.6 组件分解与每样本渲染成本

每个样本的 Cycles 渲染次数（与现方案同构，**没有变贵**）：

| 项 | 次数 |
|---|---|
| ambient | 1 |
| dark | 1 |
| 场景原生灯具分量 | 1（单灯）~3（子集） |
| 点光分量 | 16 |
| 面光分量 | 6 |
| passes（albedo/normal + 每光源 irradiance，`pass_samples=1`） | 1 + 每分量 1 |
| Workbench mask（候选筛选/可见性） | 3~9（低分辨率、可离线复用） |
| **合计** | **≈ 25 次 512² Cycles + 少量 Workbench** |

相对现方案的收益：**没有物体导入/缩放/碰撞收缩的重试**，也没有"相机候选 × mask 渲染"
的反复（新方案的候选筛选在 CPU 上做，见 3.2）。相对成本：光照分量数量若扩大（原生灯具
枚举），渲染次数线性上升，需要在 `lighting_variants_per_camera` 上取舍。

### 3.7 manifest 与 split 契约要跟着改（容易被低估）

`tokenlight/build_manifests.py` 现在硬性要求：

- `row["asset_uid"]` 存在且等于 job 的 `object_uid`（`_validate_job_metadata`）；
- `lineage.render_job_digest` 与当前 manifest 一致；
- `license.decision == "allowed"`（object 许可 + base scene 许可）；
- `split_profile ∈ {object-held-out, scene-held-out}`，且 `scene-held-out` 需要 ≥3 个场景。

新方案必须：`asset_uid → scene_id`（或保留 `asset_uid` 字段填 `scene_id` 以免下游改动）、
split 固定 `scene-held-out`、许可校验从"物体许可"改为"场景资产许可 + HDRI 许可"。
这一块**是必须动的代码**，不是配置能绕过的。

---

## 4. 已验证的原型（实证，不是推演）

### 4.1 交付物

| 文件 | 内容 |
|---|---|
| `scene_cameras.py` | 无 Blender/pydantic/yaml 依赖的相机规划原型：`FreeSpace`（AABB 自由空间 + slab 射线）、`camera_basis`/`blender_rotation_euler`（对齐 `coordinates.py` 契约）、`evaluate_pose`（6 条 gate + 打分）、`sample_camera_poses`（两级采样、确定性、间距约束）、`build_camera_job`/`build_scene_camera_jobs`（产出渲染 job manifest） |
| `tests/test_scene_cameras.py` | **14 项测试全部通过** |
| `tools/bench_scene_cameras.py` | 4625 节点 / 200 m 量级合成场景的基准脚本 |

运行方式（在 `datacompose` 的上一级目录）：

```bash
python -m unittest datacompose.tests.test_scene_cameras
python -m datacompose.tools.bench_scene_cameras --nodes 4000 --cameras 8
```

### 4.2 量化结果

测试覆盖的行为（全部通过）：

- 相机在桌子里 → `camera_inside_geometry`；
- 相机离墙 0.25 m 正对墙 → `near_geometry_in_view`（**穿模判据成立**）；
- 相机仰头 75° 只看天花板 → `single_node_dominates`；
- 房间角落 45° 斜视 → 接受，且 `hit_fraction ≥ 0.9`、`min_depth ≥ 0.3`、`spread ≥ 1.35`；
- 锚点严格落在前置射线上（`anchor = location + forward·median_depth`）；
- 同种子两次采样**逐位一致**，换种子结果不同；
- 地板策略下所有接受位姿的 z ∈ [1.10, 1.80]；
- 间距约束生效；
- 生成的 job 能被 `coordinates.CameraFrame.from_row`（含 `verify_rotation` 无 roll 校验）
  与 `Intrinsics.from_camera` 双向接受，`fx/fy/cx/cy` 正确；
- 无地板实体的场景自动退化到 `height_policy="free"`。

基准（4625 节点 / 625 个地板 box / 200 m 见方）：

| 指标 | 数值 |
|---|---|
| `FreeSpace` 构建 | 17 ms |
| 单位姿评估（24×24 = 576 射线） | 90 ms |
| 单位姿评估（12×12 = 144 射线） | 24 ms |
| 搜索 8 个相机（两级 gate） | **1.55 s → 0.19 s / 接受相机** |
| 合成box场景接受率 | 17.8%（主要拒绝：`no_geometry_in_view` 28/45，即朝向天空） |

结论：**相机搜索是秒级开销，与渲染（秒~分钟级/分量）相比完全可忽略**，因此"相机位置"
确实不再是瓶颈——瓶颈变成"如何让视角有信息量"（gate 调参）与渲染成本。

---

## 5. 成本模型

每样本 ≈ 25 次 Cycles 512²/64spp 渲染（3.6 节）。设单次渲染 `t` 秒：

```
T_sample ≈ 25 · t
样本/GPU·小时 ≈ 3600 / (25·t)
100k 样本所需 GPU 小时 ≈ 100000 · 25 · t / 3600 ≈ 694 · t
```

| 单次渲染耗时 `t` | 样本/GPU·小时 | 100k 样本 GPU·h |
|---|---|---|
| 2 s | 72 | ~1400 |
| 4 s | 36 | ~2800 |
| 8 s | 18 | ~5600 |

**这个成本与现方案逐样本相同**（现方案每 job 也是同样 25 次分量渲染），所以本方案不是
"更便宜"，而是**更可靠**：不再有 LLM 配对错误、不再有物体-场景物理不合理、不再有
碰撞收缩失败导致的整 job 报废，并且样本全是真实场景的原生光照分布。

若需要压成本，最有效的三个旋钮是：`point_lights_per_scene`（16→8）、
`samples`（64→32 + denoise）、`resolution`（512→384），以及把 irradiance passes 改为
"只在选定子集上生成"。

---

## 6. 风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| **样本信息量下降**（最主要） | 没有物体当显著性锚点，容易出现大量"空白墙/地面" | 3.2 的 `near_fraction`、`single_node_fraction`、`depth_spread` 三条 gate + 打分排序取 Top-K；再加"硬约束：视野内至少包含 N 个不同实体" |
| 视角重复 | 同一房间采样出 30 个几乎一样的机位 | `min_camera_spacing` + 每场景配额；后续可加"按视野内实体集合去重" |
| 场景原生灯不可控 | FBX 里灯具是 mesh + emissive 材质，不是 light object | `_disable_native_lighting()` 已有材质 emissive 开关；新增"只开指定材质组"的开关即可 |
| 光照"分解"语义变化 | 现状 `dark.exr` 是"全关"，原生灯组多了以后要保证可加性 | 保持"每次只开一组"的渲染范式（现方案已如此），把原生灯组当 `in_scene_lights` 的多条目 |
| 室内外/尺度混算 | Bistro 同时有室内小房间与室外街区，同一套阈值不通用 | 阈值按场景 profile 覆盖（沿用现有 `m4.scene_render_overrides` 机制）；室外放宽 `hit_fraction`、室内收紧 `near_fraction` |
| 相机贴脸但合法 | 0.3 m 外的墙面也算"无碰撞" | `near_window` + `max_near_fraction` 已覆盖；必要时对极端 FOV 用 `closeup` 直接拒绝 |
| 许可与合规 | 物体许可白名单退场，但场景/HDRI 许可仍需留痕 | `license` 块改为 `scene_license + hdri_license`，`policy_version` 继续必填；`build_manifests` 的 gate 相应改造 |
| 旧数据兼容 | 现有 `components/*/metadata.json` 与新契约不同 | 新 `job_id` 前缀（`scene_camera_`）天然隔离；`build_manifests` 用 `contract` 字段分流 |

---

## 7. 里程碑与当前落地状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| **M0** | 相机规划原型 + 测试 + 基准 | ✅ 完成：`scene_cameras.py`，14 项测试，0.19 s/相机 |
| **M1** | Blender 侧精校验：`BVHTree`/`ray_cast` 复核 AABB 通过的位姿；`clip_start` 与 `near_clearance` 对齐 | ✅ 代码完成（`blender/render_scene.py::_validate_camera_pose`），**运行时未验证**（本机无 Blender/数据） |
| **M2** | job 生成器接进 CLI + `render` 复用 worker 骨架（去掉 object 字段） | ✅ 完成：`plan_scene_cameras.py` / `render_scenes.py`；契约由 `tests/test_render_contract.py` 静态校验 |
| **M3** | 光照：HDRI 轮换、原生灯具、点光/面光范围深度自适应 | ✅ 完成（`render_scene.py` 的 `_select_native_fixture` / `_render_point_components`），可加性回归待真机 |
| **M4** | `build_manifests` 契约迁移（`scene_id`、`scene-held-out`） | ✅ 完成：`tokenlight/build_manifests.py` + `validate_components.py`，已验证 3 场景切分与坐标一致性 |

**已验证**（无 Blender 环境下可复现）：
`python -m unittest datacompose.tests.test_scene_cameras datacompose.tests.test_render_contract datacompose.tests.test_launcher` → 22 项通过；
`test_intrinsic_passes` 的 22 项在“删除后”的包副本上通过；`build_manifests` 端到端产出 3 个 split 且坐标正确。

**未验证**：真实 Cycles 渲染、合并后分量与总渲染的可加性、Bistro 规模下的实跑耗时——都需要 Ubuntu + Blender 4.5 + 数据。

---

## 8. 需要你拍板的 6 个决策点

1. **每场景相机数**：32/64/128？直接决定数据量与渲染成本（第 5 节）。
2. **光照多样性从哪来**：只用 HDRI 轮换（最省），还是必须包含场景原生灯具的部分点亮？
3. **是否需要"看向指定实体"的语义约束**（例如镜头必须包含至少 1 个 table/chair），
   还是纯粹靠 gate 的信息量筛？前者要引入"实体显著性"，后者更简单但视角偏随机。
4. **split 口径**：固定 `scene-held-out`？如果是，Bistro 的"场景"是 FBX 粒度还是"房间/街区"粒度？
   （直接影响泛化评测的可信度）
5. **是否保留 object 路线作为可选子集**：即两套 job 类型并存（`composition_*` 与
   `scene_camera_*`），还是彻底切换？并存成本低但维护面变大。
6. **是否接受"无物体"带来的显著性退化**：如果下游模型需要"主体物体"作为监督信号，
   纯场景光照数据可能不够；这一点最好在上百样本的试渲染后用人眼抽检确认。

本文档最初是可行性论证；对应代码现已落地为仓库主线（`prepare_scenes.py` → `plan_scene_cameras.py`
→ `render_dataset.py`，物体链路已移除）。第 4 节的数字可在不装 Blender、不碰数据的前提下复现；
第 8 节的决策点仍然有效——尤其"每场景相机数"和"光照多样性来源"直接决定成本与数据质量。
