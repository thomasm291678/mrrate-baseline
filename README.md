# MR-RATE: Multi-Modal MRI Report Generation

## 项目简介

本项目基于 DenseNet3D + Qwen2.5-3B-Instruct 实现多模态 MRI（T1/T2/Flair）的自动报告生成。
V3 架构引入 CVPR 2025 Spark3D 论文的多尺度 CNN + 轻量 ViT 自注意力融合方案。

---

## 目录结构

```
qi/
├── README.md              ← 本文件
├── weights/               ← 模型权重
│   ├── best_model.pt      ← V1 最终权重（最优 loss checkpoint）
│   └── last_model.pt      ← V1 最后 epoch 权重
├── code/                  ← 核心代码
│   ├── encoder.py         ← V3 Spark3D 多尺度 CNN+ViT 编码器（HybridEncoder + ReportingModel）
│   ├── train.py           ← V3 训练脚本（分组学习率 + LoRA + AMP）
│   ├── densenet3d.py      ← DenseNet3D backbone 实现（densenet3d_201）
│   ├── mrrate_dataset.py  ← MR-RATE 数据集加载器（NIfTI → Tensor）
│   ├── run.sh             ← 训练启动脚本（farm01 GPU6）
│   └── watchdog.sh        ← 自动守护进程（崩溃重启）
├── checkpoints/           ← 中间 checkpoint（如有）
└── logs/                  ← 训练日志
    ├── v1_full_train.log  ← V1 完整训练日志
    ├── v1_e9.log          ← V1 epoch 9 详情
    └── v1_e10.log         ← V1 epoch 10 详情
```

---

## 权重文件说明

### best_model.pt (9.1 GB)

**模型架构**: V1 — DenseNet3D-201 encoder + Qwen2.5-3B-Instruct (LoRA r=8)

**内容**:
- `encoder_state`: 包含 t1_enc / t2_enc / flair_enc 三个 DenseNet3D 编码器权重
- `llm_state`: Qwen2.5-3B LoRA adapter 权重
- `optimizer_state`: AdamW 优化器状态
- `epoch`: 8 / `step`: ~2500+
- `loss`: 0.277 (训练交叉熵 loss)
- `best_loss`: 0.277

**训练配置**:
- 数据集: MR-RATE (训练集)
- Epochs: 10
- Batch size: 2, Gradient accumulation: 2 (effective batch = 4)
- LR: 1e-4 (all params)
- AMP: bfloat16 mixed precision
- 编码器输出: 2048-dim → 4 visual tokens → Qwen LoRA

**用途**: V3 训练的 DenseNet backbone 权重初始化源（通过 `load_densenet_weights()` 加载）

### last_model.pt (9.1 GB)

与 best_model.pt 相同结构，epoch 10 最终 checkpoint，loss 略高于 best_model.pt。

---

## 代码文件说明

### encoder.py — V3 Spark3D 编码器

**设计理念** (源自 CVPR 2025 Spark3D):

1. **3D CNN backbone > ViT backbone** 用于医学图像
2. **多尺度特征金字塔** — 保留 U-Net 风格的 skip connection
3. **保留 MAE 预训练权重** — DenseNet3D 已在 8M+ MRI 上 MAE 预训练
4. **轻量 Self-Attention** — 仅用于跨尺度融合（2 层 TransformerBlock）
5. **不冻结 CNN** — 用小学习率微调（cnn_lr=1e-5）

**架构逐层说明**:

```
每模态 MRI 128³:
  DenseNet3D-201 (stage_channels=[256, 512, 1792, 1920])
    ├─ Stage 0: (B, 256, 32³)  → AdaptiveAvgPool(2,2,2) → Conv1×1→512 → 8 tokens
    ├─ Stage 1: (B, 512, 16³)  → AdaptiveAvgPool(2,2,2) → Conv1×1→512 → 8 tokens
    ├─ Stage 2: (B, 1792, 8³) → AdaptiveAvgPool(2,2,2) → Conv1×1→512 → 8 tokens
    └─ Stage 3: (B, 1920, 4³) → AdaptiveAvgPool(2,2,2) → Conv1×1→512 → 8 tokens

  Scale embedding → concat → 32 tokens per modality × 512-dim
  + Position embedding
  → 2-layer TransformerBlock (MultiheadAttention 8 heads)
  → LayerNorm → Linear(512→2048)

× 3 模态 = 96 visual tokens → Qwen LoRA → report generation
```

**关键类**:
- `TransformerBlock`: Pre-norm, 8-head MHA, 4× MLP ratio
- `HybridEncoder`: 单模态多尺度编码器
- `ReportingModel`: 三模态包装器 + modality embedding
- `SingleModalMAE`: V1 MAE 预训练模型（保留向后兼容）

### train.py — V3 训练脚本

**关键特性**:
- 分组学习率: CNN backbone (1e-5) / Projection + ViT + LoRA (1e-4) ← Spark3D #5
- V1 DenseNet 权重自动加载（通过 `load_densenet_weights()` 映射前缀）
- V1 Qwen LoRA 权重恢复
- AMP bfloat16 混合精度
- 梯度累积 (ga_steps=2)
- 梯度裁剪 (max_grad_norm=1.0)
- 线性 warmup 学习率调度
- 自动保存 best_model / last_model / checkpoint

**运行方式**:
```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
    --v1_ckpt outputs/report_gen/best_model.pt \
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
    --batch_size 2 --ga_steps 2 --epochs 5 \
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \
    --vit_dim 512 --vit_heads 8 --vit_depth 2 --use_amp
```

### densenet3d.py — DenseNet3D Backbone

- 实现 DenseNet3D-201 (block_config=[6,12,48,32])
- 支持 `return_stages=(0,1,2,3)` 多级特征提取
- `stage_channels = [256, 512, 1792, 1920]`

### mrrate_dataset.py — 数据集加载

- 从 NIfTI (.nii.gz) 加载 T1/T2/Flair 模态
- 自动归一化、resize 到 128³
- 处理缺失模态（mask + 零填充）

### run.sh — 训练启动脚本

在 farm01 GPU6 上启动 V3 训练。所有超参数已预设。

### watchdog.sh — 守护进程

每 60 秒检查训练进程，崩溃自动重启。

---

## 训练环境

| 项目 | 详情 |
|------|------|
| 训练服务器 | farm01 (10.176.60.71) GPU6 |
| GPU | NVIDIA GeForce RTX 4090 48GB |
| Python 环境 | /home/jiaqigu/hidnet_env (Python 3.12, PyTorch 2.x) |
| 数据集路径 | /mnt/nas1/disk07/public/mr_data/MR-RATE |
| Qwen 模型路径 | /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct |

---

## V1 → V3 演进总结

| 组件 | V1 (old) | V3 (new) |
|------|----------|----------|
| 编码器 | DenseNet 单级 (Stage 3 only) → AdaptiveAvgPool(4³) → Flatten(141312) → Linear(2048) | DenseNet 四级多尺度 (Stages 0-3) → Conv1×1 投影 → 32 tokens × 512-dim × 3 mod |
| 空间建模 | 无 (纯 CNN 全局池化) | 2-layer TransformerBlock Self-Attention 跨尺度融合 |
| 参数量 | ~141K tokens flatten 后全连接 | 96 tokens 轻量 Self-Attn + 投影 |
| CNN 微调 | 冻结 (lr=0) | 微调 (lr=1e-5) |
| 理论基础 | 传统 CNN-LoRA-LLM | Spark3D CVPR 2025 多尺度 + ViT 融合 |
| Visual tokens | 4 (单级压缩) | 96 (32 per modality, 跨尺度) |

---

## 备份日期

2026-07-12

## 联系人

jiaqigu
