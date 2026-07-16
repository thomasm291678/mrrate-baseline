# MR-RATE: Multi-Modal MRI Report Generation

> **任务**：脑部 MRI 三模态（T1 / T2 / FLAIR）→ 放射学报告自动生成  
> **核心 LLM**：Qwen2.5-3B-Instruct（LoRA 微调）  
> **当前版本**：V6 — 120 CNN tokens 直灌 Qwen，两阶段训练

---

## 项目结构

```
5555/
├── README.md
├── .gitignore
│
├── v6/                                ← 当前版本：两阶段训练
│   ├── model.py                       ← MRRCNN + MultiModalEncoder（自包含，无外部依赖）
│   ├── train_phase1.py                ← Phase 1: 对比学习预训练 encoder
│   ├── train_phase3.py                ← Phase 3: Qwen LoRA 报告生成
│   └── mrrate_dataset.py              ← 数据集加载器
│
├── v5_mrrcnn/                         ← V5 留存（Phase 2 实验）
│   ├── encoder_v5.py                  ← V5 编码器（含 DiseaseAwareProjector）
│   ├── train_v5_phase2.py             ← Phase 2: Disease-Aware Projector 对齐
│   └── mrrate_dataset.py
│
├── v4_uniformer/                      ← V4: BrainMVP UniFormer
│   ├── uniformer_blocks.py
│   ├── encoder_v4.py
│   ├── train_v4.py
│   └── generate_init.py
│
├── v1_v3/                             ← V1-V3: DenseNet3D + Spark3D
│   ├── densenet3d.py
│   ├── single_modal_encoder.py
│   ├── merge_lightweight.py
│   ├── encoder.py
│   ├── train_report_gen.py
│   └── mrrate_dataset.py
│
├── docs/                              ← 项目文档
│   ├── PROJECT_README.md
│   ├── V5_项目架构与技术原理.md
│   ├── V5_项目总结.md
│   ├── V5_Phase3_根因分析报告.md
│   └── FedAvg_MRCNN_操作手册.md
│
├── evaluation/                        ← 评估框架
│   ├── evaluation_v2.py
│   ├── eval_llm.py / eval_report.py
│   └── ...
│
└── scripts/                           ← 运维工具
```

---

## 版本演进路线

```
V1 (2025-Q4)    DenseNet3D-201 ×3, 单级全局池化, 4 visual tokens
       │         参数量大，信息压缩过度
       ▼
V3 (2026-Q2)    Spark3D 多尺度, 96 tokens (32×3mod), Transformer 跨尺度融合
       │         显存 >29GB
       ▼
V4 (2026-Q2)    BrainMVP UniFormer, 120 tokens (40×3mod), CNN+Attention 混合
       │         3D 展平后 attention token 爆炸, 训练慢
       ▼
V5 (2026-Q3)    MRRCNN 纯 CNN + SE3D + StageAttention, 三阶段渐进训练
       │         显存 9GB, 0.37s/step
       │         DiseaseAwareProjector 37→120 信息瓶颈
       ▼
V6 (2026-Q3)    120 CNN tokens 直灌 Qwen, 两阶段训练
                 无中间压缩层，信息保真度最高
```

### 核心指标对比

| 维度 | V3 (Spark3D) | V4 (UniFormer) | V5 (MRRCNN) | **V6 (MRRCNN v2)** |
|------|:---:|:---:|:---:|:---:|
| Backbone | DenseNet3D ×4stages | UniFormer | MRRCNN 6-stage | MRRCNN 6-stage |
| Visual tokens | 96 | 120 | 37 (Projector) | **120 (raw CNN)** |
| 训练阶段 | 2 | 1 | 3 | **2** |
| 训练显存 | 22GB | 29GB | 9GB | **16GB** |
| 信息瓶颈 | 无 | Attention | 37 disease tokens | **无** |

---

## V6: 当前版本

### 架构

```
MRI (T1/T2/Flair)
  │
  ▼
MRRCNN  ×3     ← 6 级 CNN 金字塔 + SE3D + StageAttention
  │              每模态 40 × 512-dim tokens
  ▼
Linear(512→2048) + mod_emb  ← 投影到 LLM 空间
  │
  ▼
[120 × 2048] visual tokens   ← 三模态拼接，直灌 Qwen
  │
  ▼
Qwen2.5-3B-Instruct (LoRA r=8)  → 放射学报告
```

### 两阶段训练

```
Phase 1: Encoder 对比学习（无需任何外部模型）
  ├── 三模态各自 MRRCNN → ContrastiveHead(512→256)
  ├── 同患者 T1/T2/FLAIR 互为正样本
  ├── 冻结投影层 (t1_proj/t2_proj/flair_proj)，只训练 backbone
  └── 输出: phase1_latest.pt（encoder_state）

Phase 3: Qwen LoRA 报告生成
  ├── 冻结 encoder → 加载 Phase 1 encoder_state
  ├── 120 visual tokens 作为前缀注入 Qwen
  ├── LoRA r=8, target_modules=[q_proj,k_proj,v_proj,o_proj]
  └── 自回归生成报告
```

### 与 V5 的区别

| | V5 | V6 |
|---|-----|-----|
| Phase 2 | DiseaseAwareProjector 对齐 Qwen embedding | **已删除** |
| Phase 3 输入 | 37 compressed tokens | **120 raw CNN tokens** |
| 信息损失 | 120→37 压缩 | **无，直灌** |
| Encoder 文件 | `encoder_v5.py` 含 Projector | `v6/model.py` 纯 encoder |

### 为什么砍掉 Phase 2

Phase 2 的 DiseaseAwareProjector 用 37 个可学习 disease query 把 120 个 CNN token 压缩到 37 维——虽然在疾病维度上可解释，但比 120 直通的信息量少了 3 倍以上。V6 直接扔掉这层压缩，让 Qwen 自己从 120 个原始 token 中学习注意力模式。

---

## 训练命令

```bash
# Phase 1: Encoder 对比学习
cd v6
python train_phase1.py \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --log_dir ../outputs/report_gen \
  --batch_id batch27 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-4

# Phase 3: Qwen LoRA 报告生成
python train_phase3.py \
  --encoder_ckpt ../outputs/report_gen/phase1_latest.pt \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
  --log_dir ../outputs/report_gen \
  --batch_id batch27 \
  --epochs 3 \
  --batch_size 8 \
  --lr 5e-5
```

---

## 环境要求

| 组件 | 详情 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.x (CUDA 12.8) |
| GPU | ≥24GB (推荐 RTX PRO 6000 / 98GB) |
| 数据集 | MR-RATE (NAS: `/mnt/nas1/disk07/public/mr_data/MR-RATE`) |
| Qwen | Qwen2.5-3B-Instruct |
| transformers, peft, safetensors | pip install |

---

## 联系人

jiaqigu
