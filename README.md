# datacompose — 纯场景数据合成

把**已有场景**（Bistro 等 FBX）在**可控光照**下渲染成 TokenLight 需要的线性分量、内禀通道与 manifest。

不再插入 Objaverse 物体，也没有物体/场景的语义配对：样本 = `(scene, camera pose, lighting)`。

## 三步跑完

```bash
conda activate <env>
pip install "pydantic>=2" pyyaml numpy        # 渲染侧还需 Blender 4.5 + NVIDIA GPU

cd <本项目目录>                                # 就是包含这些脚本的目录

# 1) FBX → 归一化 .blend + scene.json
python prepare_scenes.py --blender-bin /opt/blender-4.5/blender

# 2) 在每个场景里规划"视野无穿插"的相机
python plan_scene_cameras.py --cameras-per-scene 32

# 3) 按相机 + 光照渲染 TokenLight 分量
python render_dataset.py --blender-bin /opt/blender-4.5/blender --workers 2
```

失败重跑时，用 `--clear-failed` 一次性清掉本批失败留下的 partial（只清带 `failure.json`、且没有已完成产物的那些）：

```bash
python render_dataset.py --clear-failed --blender-bin /opt/blender-4.5/blender
```

只想清某一个 job 就用 `python clear_render_partial.py --job-id <job_id>`。

**重试前需要删什么**：只需要清 `outputs/tokenlight_dataset/components/*.partial`。`outputs/` 下其余内容可以留着——
`scene_camera_jobs.jsonl` 是规划结果（删了要重跑 `plan_scene_cameras.py`），`render_errors.jsonl` /
`render_summary.json` 是上一次的诊断记录（每次跑都会覆盖）。已完成的 `components/<job_id>/` 也会被复用。

| 脚本 | 作用 | 主要输出 |
|---|---|---|
| `prepare_scenes.py` | FBX → 归一化 `.blend` + 场景图 | `paths.scene_output` |
| `plan_scene_cameras.py` | 场景内自由空间相机采样（≥30% 拒绝率正常） | `paths.render_jobs_output` |
| `render_dataset.py` | 调 Blender 渲染分量 + metadata | `paths.tokenlight_output` |
| `clear_render_partial.py` | 人工确认后清理失败 partial | — |

### 配置

脚本默认读 `configs/default.yaml`；也可以 `--config my.yaml`，或只设一次环境变量：

```bash
export DATACOMPOSE_CONFIG=/data/frankie/datacompose-sceneonly/sceneonly.yaml
python prepare_scenes.py --blender-bin /opt/blender-4.5/blender
```

`--config` 优先于环境变量。`paths.scene_cache` 必须位于 `project.root` 之内（场景片段以相对路径记录）。
**不再需要 `OBJECT_ROOT`**——代码里没有任何物体资产。

## 相机是怎么定的

`scene_cameras.py`（纯 numpy，不需要 Blender）在场景里采样位姿，并用 `scene.json` 的 node AABB
做宽相判据：

- 相机不在几何体内（`camera_radius`）；
- 视锥射线最小命中深度 ≥ `near_clearance` —— **这就是"视野里不穿模"**；
- 命中率、近景占比、单节点占比、深度跨度（p90/p10）限制"对着空处/对着墙"。

通过的位姿再由 `blender/render_scene.py` **用真实三角面重新校验一遍**（`scene.ray_cast` + `BVH`），
不通过就整个 job 失败，而不是产出一张穿模的图。判据阈值在 `m4.camera`（规划）与 `m4.camera_gate`
（渲染复核）里，两边保持一致。

### 目标条件化取景（`entity_conditioned_v1`）

上面这套判据有个副作用：**两面墙组成的墙角是它的最优解**（没有单一节点占优、深度跨度又大），
于是会产出"画面里只有墙角、没有物体"的样本。把 `m4.camera.strategy` 切成
`entity_conditioned_v1` 后，`scene_targets.py` 会先挑出可成景的实体，再在它周围的球壳上取景，
并要求：

- 目标筛选是**几何判据 + 打分排序**：硬过滤扁平度/厚度/最长轴/占场景跨度（挡掉街道、门垫、
  地形、建筑壳体），再用 `framability_score` 排序。category 只做早期剔除和排序惩罚——分类器
  输出不够可靠，不能当白名单。
- `target_visible_ratio`：目标表面样本有多少真正看得见（AABB 射线，带接触容差，
  所以"杯子放在桌上"不算被遮挡）；
- `target_screen_ratio`：目标投影包围框占画面的比例，太大是糊屏特写；
- 目标中心必须落在画面安全区内；
- 旧的近裁剪/命中率判据仍然生效，但只统计**非目标**几何，目标本身可以占满画面。

同时保留 `mix_with_free_space`（默认 0.30）比例的远景，远景也必须含一个真正可见的目标，因此
不会再出现纯墙角。所有阈值都在 `m4.camera.target` 下，可只改配置、不改代码。

注意绕行取景的机位间隔是**角度**（`target.min_azimuth_separation_degrees`），不是欧氏距离
（`min_camera_spacing`）：小目标在 0.6 m 轨道上任意两个机位最多相距 1.2 m，用 1.5 m 的间距会让
一个目标只能出一个机位。

### 室内场景：空心容器模型（`m4.camera.container`）

进口的室内片段常常是**一整块网格**（单个节点，AABB 覆盖整个房间）。在"盒子=实心"的自由空间
模型里，这种场景里每一个眼高点都算"在几何体内部"，从室内发出的射线也打不到壳体，于是整个房间
一个相机都规划不出来。`container_v1` 把这种盒子识别为**空心**：`clearance` 变成"到最近墙面的距离"、
射线命中的是**对面墙**，并且只在相机位于容器内时，容器的命中不计入"单一物体占满画面"。

容器判据纯几何（三轴都 ≥ `min_extent`、体积 ≥ `min_volume`、AABB 内含其它节点中心），不看
category。把 `m4.camera.container.model` 设成 `aabb_legacy` 即恢复旧的实心盒模型，用来复现旧数据集。
室内地板通常就在壳体网格里，所以地板来源优先级是：floor 类实体 → 容器底面 → 高度带 + 绝对值钳制
（`m4.camera.free_height_clamp`）。

## 光照

- 环境光：`m4.render.hdri_root` 指向 `.hdr/.exr` 目录（每个样本确定性抽一张），留空则用常色；
- 场景原生灯具：实体类别命中 `m4.fixture_categories`（lamp/lantern/...）且画面内可见时，用它当发光体；
  没有可用灯具时在镜头前按深度比例放一个程序化面光源；
- 点光/面光：`m4.render.point_lights_per_scene` 等，偏移量都是**帧内深度的小数**，随房间尺度自动放大。

## 依赖

| 用途 | 包 |
|---|---|
| 配置 / 数据协议 | `pyyaml`、`pydantic>=2` |
| 相机规划、内禀通道 | `numpy` |
| 数据集校验（可选） | `tqdm`、`opencv-python` |
| 渲染 | 外部 **Blender 4.5**（子进程调用，不用 pip 的 `bpy`）+ NVIDIA 驱动 |

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

`tests/test_scene_cameras.py` 覆盖相机判据与确定性；`tests/test_scene_camera_targets.py` 覆盖目标条件化
取景（墙角场景产不出相机、目标可见性/画面占比、遮挡与接触容差、混采分配）；
`tests/test_scene_containers.py` 覆盖空心容器模型（容器检测、室内 clearance/raycast、`aabb_legacy`
复现、地板来源优先级、目标几何过滤与排序、轴平行射线回归）以及配置键与代码字段的一致性；
`tests/test_launcher.py` 覆盖脚本化运行方式；
`tests/test_coordinates.py`、`test_manifest_camera.py`、`test_intrinsic_passes.py` 覆盖坐标/分量契约。

## 数据集清单（可选）

```bash
python tokenlight/build_manifests.py    --config stage2.yaml   # 按 scene 切 train/val/test
python tokenlight/validate_components.py --config stage2.yaml   # 分量完整性与坐标一致性
```

split 固定按 `scene_id`（scene-held-out），一个场景不会跨 split。

## 其他文档

- `GUID.md`：坐标约定、manifest 契约、内禀通道与材质判定；
- `docs/scene_only_render_feasibility.md`：纯场景方案的取舍、成本模型与风险；
- `scripts/remove_legacy_object_pipeline.sh`：清掉残留的 object 时代文件（`--apply` 执行）。
