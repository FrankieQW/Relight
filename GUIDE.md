# FLUX.1-Kontext-dev LoRA + Lighting Tokens 指南

## 1. 方法概览

Kontext 原生是 instruction-based image editing。本工程只保留它的图像编辑骨干，完全移除
文本条件；模型仅接收 source image、带噪 target 和结构化 lighting tokens。

每条训练样本包含四部分：

```text
condition image = TokenLight source image
lighting tokens = 连续数值、known/valid、字段类型和任务类型
target image    = TokenLight 合成的监督图像
```

训练参数分为：

- FLUX transformer 上的 LoRA 参数；
- 新增 LightingTokenEncoder 的全部参数。

FLUX 原始权重和 VAE 均冻结。因此这是“FLUX LoRA + 条件编码器”微调，不是只
保存 LoRA 的纯 LoRA 方法。

## 2. 独立工程结构

```text
new_method/
├── GUIDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── train.yaml
├── scripts/
│   ├── check.sh
│   ├── train.sh
│   └── infer.sh
├── src/tokenlight_flux/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── lighting.py
│   ├── train.py
│   └── infer.py
├── train.py
└── infer.py
```

建议把整个目录复制到：

```text
/mnt/afs_fangwenqi/new_method
```

工程不导入 `Lumina-T2X`、旧 TokenLight 包或 BFL `flux` 源码。仍需保留的外部内容只有
TokenLight 数据集和 FLUX 权重。Blender、vLLM、Lumina NextDiT、旧 SDXL VAE 与旧
lighting encoder 都不需要复制。

## 3. 权重位置

在 Hugging Face 接受 `black-forest-labs/FLUX.1-Kontext-dev` 的 gated 模型条款后，将
完整 Diffusers snapshot 下载到：

```text
/mnt/afs_fangwenqi/models/FLUX.1-Kontext-dev
```

至少包含：

```text
FLUX.1-Kontext-dev/
├── model_index.json
├── scheduler/
├── transformer/
├── vae/
├── text_encoder/
├── text_encoder_2/
├── tokenizer/
└── tokenizer_2/
```

不要只下载 BFL 原生格式的 `flux1-kontext-dev.safetensors` 和 `ae.safetensors`。所有加载
都设置 `local_files_only=True`，代码不会自动下载缺失文件。

FLUX.1-Kontext-dev 使用其模型页面声明的非商业许可；模型产物和数据资产仍须遵守各自
许可约束。

## 4. 数据与四类任务

默认读取：

```text
/mnt/afs_fangwenqi/data/tokenlight_1000obj_960_in_scene/
├── components/
└── manifests/
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

数据层直接读取 manifest，不依赖旧配置。合成仍在线性 RGB 中完成：

```text
ambient_scale  = dark + (ambient - dark) * scale
global_diffuse = ambient + diffuse_component - dark
add_light      = ambient + max(point_component - dark, 0) * color * intensity
in_scene_light = ambient + max(fixture_on - dark, 0) * color * intensity * transition
```

随后执行 exposure、Reinhard tone mapping、中心裁剪和缩放，输入 VAE 的范围为 `[-1,1]`。
同一个 `(split, index)` 总是使用同一 seed，因而得到同一 source/target/control。

## 5. Lighting token 结构

`model.max_lights=3` 时共有 34 个 scalar tokens：

```text
ambient                                      1
global_diffuse                               1
3 × (slot_valid + xyz + rgb + intensity + softness) 27
in_scene (rgb + intensity + transition)      5
总计                                         34
```

每个 scalar token 包含：

- 连续值的 Gaussian Fourier `sin/cos` 特征；
- `known` 位；
- `valid` 位；
- learned field embedding，用于区分 ambient、x、r、intensity 等字段；
- learned task embedding；
- learned lighting-type embedding。

这些 token 由 MLP 投影到 FLUX `joint_attention_dim`。文本 context 长度固定为 0，因此
它们就是 FLUX context stream 中的全部 token。joint attention 中的区分方式是：

```text
lighting token        数值 embedding + field/task/type embedding
target image token    image ID 第一轴 = 0
condition image token image ID 第一轴 = 1
lighting token        position ID 第一轴 = 2，第二轴 = scalar token index
```

模型不接收自然语言。任务类型由 learned task embedding 表示，连续数值只来自 lighting
tokens。

fixture mask 当前不输入模型。`in_scene_light` 的二维灯具位置来自 condition image 中可见的
灯具，颜色、强度和 transition 来自 lighting tokens。若必须显式输入 mask，需要再增加
mask adapter。

## 6. 创建独立环境

不要修改旧 `lum` 环境：

```bash
conda create -n flux-kontext python=3.11 -y
conda activate flux-kontext
cd /mnt/afs_fangwenqi/new_method

# 先按服务器 CUDA/驱动安装匹配的 PyTorch，再安装其余依赖：
pip install -r requirements.txt
pip install -e .
```

## 7. 配置

唯一持久配置是 `configs/train.yaml`。默认值：

- 960×960；
- BF16；
- FLUX LoRA rank/alpha = 16/16；
- 34 个 lighting tokens；
- 每个 scalar 使用 32 个 Fourier frequencies；
- LightingTokenEncoder hidden dimension = 512，再投影到 FLUX context dimension；
- 每卡 micro batch = 1；
- gradient accumulation = 4；
- 8 卡 global batch = `1 × 4 × 8 = 32`；
- gradient checkpointing 开启；
- 每 500 optimizer steps 保存 checkpoint。

正式配置保持 960。首次部署时可复制一份配置并临时改为 512 完成 smoke；确认显存和训练
链路后再使用 960。scheduler 会按实际 target token 数计算 dynamic-shifting `mu`。改变分辨率
或 lighting schema 时应创建新 run，不要恢复旧 optimizer 状态。

## 8. 服务器检查与单卡 smoke

检查环境、权重契约和真实数据：

```bash
cd /mnt/afs_fangwenqi/new_method
conda activate flux-kontext
bash scripts/check.sh
```

输出应同时包含 `lighting_token_count`、`lighting_known_count` 和 `lighting_valid_count`。

单卡执行 2 个 optimizer steps：

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 \
  bash scripts/train.sh --max-steps 2
```

`checkpoint-2` 必须至少包含：

```text
pytorch_lora_weights.safetensors
lighting_encoder.safetensors
lighting_config.json
optimizer/state/RNG files written by Accelerate
```

再检查恢复：

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 \
  bash scripts/train.sh --max-steps 4 --resume latest
```

## 9. 八卡训练

```bash
cd /mnt/afs_fangwenqi/new_method
bash scripts/train.sh
```

修改设备或端口：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 MASTER_PORT=29631 \
  bash scripts/train.sh
```

输出默认位于：

```text
/mnt/afs_fangwenqi/new_method/outputs/flux_kontext_tokenlight_lora/
├── config.yaml
├── checkpoint-500/
├── checkpoint-1000/
└── final/
```

恢复使用 `--resume latest`，或指定完整 `checkpoint-N` 目录。恢复时会严格检查 lighting
schema、Fourier 设置、context dimension 和权重 shape。

## 10. 推理

### Ambient scale

```bash
bash scripts/infer.sh \
  --lora /mnt/afs_fangwenqi/new_method/outputs/flux_kontext_tokenlight_lora/final \
  --source /path/to/component/ambient.exr \
  --output /mnt/afs_fangwenqi/tokenlighttest/ambient_05.png \
  --task ambient_scale \
  --ambient-scale 0.5
```

### Global diffuse

```bash
bash scripts/infer.sh \
  --lora /path/to/final \
  --source /path/to/source.exr \
  --output /path/to/result.png \
  --task global_diffuse \
  --source-diffuse 0.2 \
  --target-diffuse 1.0
```

### Add light

一个 `--light` 的顺序是 `x,y,z,r,g,b,intensity,softness`：

```bash
bash scripts/infer.sh \
  --lora /path/to/final \
  --source /path/to/source.exr \
  --output /path/to/result.png \
  --task add_light \
  --light '0.2,-0.1,0.8,1.0,0.7,0.4,1.2,0.1'
```

### In-scene light

```bash
bash scripts/infer.sh \
  --lora /path/to/final \
  --source /path/to/source.exr \
  --output /path/to/result.png \
  --task in_scene_light \
  --fixture-rgb '1.0,0.8,0.5' \
  --fixture-intensity 1.0 \
  --fixture-transition 1.0
```

推理只接收任务类型及对应的结构化数值参数。输出旁会生成 `.png.json`，记录 control、
lighting schema 和推理设置。

## 11. 训练计算路径

```text
source RGB  ──FLUX VAE──> condition latent ───────────────┐
                                                          ├─ Kontext image sequence
target RGB  ──FLUX VAE──> target latent ─加噪─────────────┘

numeric controls ─LightingTokenEncoder─> lighting context ─┐
                                                           ├─ joint attention
time + zero pooled condition + guidance embedding ─────────┘

FLUX transformer + LoRA → target velocity prediction
loss = MSE(prediction, noise - clean_target_latent)
```

loss 只在 target image token 部分计算。LightingTokenEncoder 和 LoRA 同时接收梯度；冻结的
FLUX 主干和 VAE 不更新。CLIP/T5 不参与前向计算。

## 12. 验收边界

当前工作区没有运行服务器 Python、GPU 或真实权重检查。正式训练前必须依次通过：

1. `scripts/check.sh`；
2. 单卡 2-step smoke；
3. checkpoint 恢复到 step 4；
4. 单样本推理；
5. 八卡正式训练。

若服务器报错，请保留完整 traceback，以及 `torch`、`diffusers`、`transformers`、`peft`、
`accelerate` 的版本信息。
