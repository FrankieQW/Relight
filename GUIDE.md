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

fixture mask 现在经过与图像一致的中心裁剪和 resize，再从 960×960 压缩为 15×15
网格。每个网格 token 包含 average occupancy、max occupancy 和 fixture-present 位，经
MLP 投影到 FLUX context dimension。它的位置 ID 为 `[3, y, x]`，因此 joint attention
可以把灯具二维位置与 target/source image tokens 对齐。非 `in_scene_light` 样本输入全零
mask 和 `fixture_present=false`。

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
- gradient checkpointing 默认关闭（显存不足时可在配置中开启）；
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
fixture_mask_encoder.safetensors
lighting_config.json
optimizer/state/RNG files written by Accelerate
```

fixture-mask 编码器是新增的可训练模块，因此启用它后不要用旧版
`flux_kontext_tokenlight_lora_no_text` checkpoint 做完整断点续训。默认配置使用新的
`flux_kontext_tokenlight_lora_fixture_mask` 运行目录并从基础模型重新开始；新格式
checkpoint 之间可以正常续训。

再检查恢复：

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 \
  bash scripts/train.sh --max-steps 4 --resume latest
```

## 9. 多卡训练

训练包装器采用 `trainable_parameters_only` DDP：每个 rank 从同一个本地 checkpoint 独立加载
冻结 FLUX 主干，DDP 只注册并同步 LoRA 和 LightingTokenEncoder。这样不会在初始化时广播
数百 MiB 的冻结权重。启动日志应包含：

```json
{"ddp_mode":"trainable_parameters_only","ddp_parameter_count":...}
```

```bash
cd /mnt/afs_fangwenqi/new_method
bash scripts/train.sh
```

修改设备或端口：

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 MASTER_PORT=29631 \
  bash scripts/train.sh
```

对于无 GPU P2P 的双 RTX 6000D，启动脚本默认设置
`NCCL_CUMEM_HOST_ENABLE=0` 和 `NCCL_IB_DISABLE=1`。手动执行 `accelerate launch` 时应先设置：

```bash
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
```

`scripts/train.sh` 会在多卡训练前先运行一个仅包含单元素
`all_reduce` 的 NCCL 预检。如果该预检也出现 CUDA 700，故障位于当前
PyTorch/CUDA/NCCL/驱动或双卡拓扑，而不是 FLUX、LoRA 或数据集代码；此时脚本会在
加载模型前停止。`NCCL_SHM_DISABLE=1` 会牺牲通信速度换取更保守的本机 Socket
传输。确认共享内存通信稳定后，可以显式设置 `NCCL_SHM_DISABLE=0` 恢复性能。

输出默认位于：

```text
/mnt/afs_fangwenqi/new_method/outputs/flux_kontext_tokenlight_lora_fixture_mask/
├── config.yaml
├── tensorboard/
├── checkpoint-500/
├── checkpoint-1000/
└── final/
```

恢复使用 `--resume latest`，或指定完整 `checkpoint-N` 目录。恢复时会严格检查 lighting
schema、Fourier 设置、context dimension 和权重 shape。

训练终端只在 local main process 显示一个 tqdm 进度条，包含 loss、学习率、梯度范数、当前
任务和单卡峰值显存。TensorBoard 每个 optimizer step 记录：

```text
train/loss
train/learning_rate
train/grad_norm
train/timestep_mean
train/sigma_mean
system/max_memory_allocated_gib
system/max_memory_reserved_gib
```

另开终端启动 TensorBoard：

```bash
tensorboard \
  --logdir /mnt/afs_fangwenqi/new_method/outputs/flux_kontext_tokenlight_lora_fixture_mask/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

### Validation

训练会从 `paths.validation_manifest` 构建独立的 validation DataLoader，并使用固定
VAE mode、固定噪声与固定 timestep 进行可重复比较：

```yaml
train:
  validation_steps: 500
  validation_batches: 8
  validation_seed: 12345
```

多卡时 `validation_batches` 是每个 rank 最多处理的批次数，损失会跨 rank 汇总。
TensorBoard 会记录 `validation/loss` 和四种任务各自的 validation loss。
启用 fixture mask 后还会记录 `validation/in_scene_mask_ablated_loss` 和
`validation/in_scene_mask_ablation_gap`。后者等于“零 mask 损失减正确 mask 损失”；持续
大于 0 才说明模型确实利用了 mask 空间条件。

当 `validation/loss` 严格低于历史最佳值时，训练会更新：

```text
<run_dir>/best_checkpoint/
<run_dir>/best_validation.json
```

`best_checkpoint` 保存 LoRA、Lighting Encoder、优化器和随机状态，可用于推理，也可将
`paths.resume_checkpoint` 指向该目录继续训练。它不会被
`checkpoints_total_limit` 的普通 checkpoint 清理逻辑删除。

## 10. 推理

### Ambient scale

专用 ambient 推理入口只需要输入 ambient EXR、LoRA 目录和输出路径：

```bash
python infer_ambient.py \
  /path/to/component/ambient.exr \
  /path/to/best_checkpoint \
  /path/to/ambient_sweep.png
```

输出是无文字、无间隔的六图横向拼接，顺序为原始 ambient、scale 0、0.375、
0.75、1.125、1.5。五次生成使用相同 seed，并且模型只加载一次。可选参数包括
`--steps`、`--guidance-scale`、`--seed`、`--cpu-offload` 和 `--config`。

要查看训练数据管线实际构造的对应 GT（不经过模型），运行：

```bash
PYTHONPATH=src python visualize_ambient_training_sweep.py \
  /path/to/component/ambient.exr \
  /path/to/ambient_training_sweep.png
```

脚本默认读取 ambient 同目录的 `dark.exr`，也可用 `--dark` 显式指定。输出顺序同样是
ambient、scale 0、0.375、0.75、1.125、1.5，并严格复现训练中的
`dark + (ambient-dark) * scale`、exposure、中心裁剪、Reinhard 和 resize。

```bash
bash scripts/infer.sh \
  --lora /mnt/afs_fangwenqi/new_method/outputs/flux_kontext_tokenlight_lora_fixture_mask/final \
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
  --fixture-mask /path/to/fixture_000_mask.png \
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
