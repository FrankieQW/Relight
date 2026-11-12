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

失败重跑时用 `python clear_render_partial.py --job-id <job_id>` 明确清掉某次渲染的 partial。

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

`tests/test_scene_cameras.py` 覆盖相机判据与确定性；`tests/test_launcher.py` 覆盖脚本化运行方式；
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
