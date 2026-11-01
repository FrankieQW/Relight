# datacompose — 独立数据合成项目

把 Objaverse 单物体资产放进/替换到 Bistro 场景，渲染 TokenLight 需要的线性光照分量、内禀通道与 manifest。

本目录是**自包含项目**：代码、`configs/`、`tests/`、`transfer.py`（坐标参考实现）都在这里，
不依赖上级仓库的 `src/`、`scripts/`、`Lumina-T2X/`。

## 快速开始

在**包含 `datacompose/` 的上一级目录**运行（`--config` 默认即 `datacompose/configs/default.yaml`）：

```bash
python -m datacompose.cli prepare-objects
python -m datacompose.cli prepare-scenes    --blender-bin /opt/blender-4.5/blender
python -m datacompose.cli prepare-geometry
python -m datacompose.cli build-render-jobs
python -m datacompose.cli render            --blender-bin /opt/blender-4.5/blender
```

## 配置

`configs/default.yaml` 的 `project.root: ..` 即本目录，`paths.*` 相对本项目根；数据/输出默认落在本目录内，
可用自定义 `--config` 指向外部数据目录。

## 测试

```bash
python -m unittest discover -s datacompose/tests -p 'test_*.py'
```

需要 PyYAML；`datacompose/tests/blender_composition_focus.py` 需要 Blender。

## 文档

坐标约定、manifest 契约、内禀通道（albedo / irradiance / normal）与三层材质判定见 `GUID.md`。
