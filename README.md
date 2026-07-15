# MR-RATE: Multi-Modal MRI Report Generation

> **任务**：脑部 MRI 三模态（T1 / T2 / FLAIR）→ 放射学报告自动生成  
> **核心 LLM**：Qwen2.5-3B-Instruct（LoRA 微调）  
> **演进路线**：V1 → V4 → V5，逐步从重骨架走向轻量高效

---

## 项目结构

```
5555/
├── README.md                          ← 本文件（总纲领）
├── .gitignore
│
├── docs/                              ← 项目文档
│   ├── PROJECT_README.md              ← 原始 V1 项目说明
│   ├── V5_项目架构与技术原理.md        ← V5 架构详解
│   ├── V5_项目总结.md                  ← V5 开发总结
│   ├── V5_Phase3_根因分析报告.md        ← Phase 3 问题诊断
│   └── FedAvg_MRCNN_操作手册.md        ← FedAvg 实验指南
│
├── v1_v3/                             ← V1-V3: DenseNet3D + Spark3D 多尺度
│   ├── densenet3d.py                  ← DenseNet3D-201 backbone
│   ├── single_modal_encoder.py        ← V1 MAE 单模态预训练
│   ├── merge_lightweight.py           ← V2 三模态轻量合并
│   ├── encoder.py                     ← V3 Spark3D CNN+ViT 多尺度编码器
│   ├── train_report_gen.py            ← Qwen 对齐 + LoRA 训练
│   └── mrrate_dataset.py              ← MR-RATE 数据集加载器
│
├── v4_uniformer/                      ← V4: BrainMVP UniFormer
│   ├── uniformer_blocks.py            ← UniFormer backbone（CNN+Attention混合）
│   ├── encoder_v4.py                  ← V4 三模态编码器
│   ├── train_v4.py                    ← V4 训练脚本
│   └── generate_init.py               ← 权重初始化工具
│
├── v5_mrrcnn/                         ← V5: MRRCNN（当前版本）
│   ├── encoder_v5.py                  ← V5 编码器（MRRCNN + DiseaseAwareProjector）
│   ├── train_v5.py                    ← Phase 1: 对比学习
│   ├── train_v5_phase1.py             ← Phase 1 正式脚本
│   ├── train_v5_phase2.py             ← Phase 2: Projector 对齐
│   ├── train_v5_phase3.py             ← Phase 3: Qwen LoRA
│   ├── merge_encoders.py              ← Phase 1 多GPU合并
│   ├── run_fedavg.py                  ← FedAvg 实验
│   └── mrrate_dataset.py              ← 数据集加载器
│
├── evaluation/                        ← 评估框架
│   ├── evaluation_v2.py               ← 主评估入口
│   ├── evaluation_v1_jss/             ← V1 评估 (JSS 标准)
│   ├── v2/                            ← V2 评估
│   ├── eval_llm.py / eval_report.py   ← 推理 + 报告评估
│   ├── eval_runner.py                 ← 批量评估
│   ├── gen_tables.py                  ← 37 类临床标签生成
│   ├── compare_llm_vs_keyword.py      ← 关键词 vs LLM 对比
│   ├── sync_eval_results.py           ← 结果同步脚本
│   └── test_radbert.py                ← RadBERT 基线实验
│
└── scripts/                           ← 运维工具
    ├── check_all.py / check_server.py  ← 服务器状态检查
    ├── check_training.py               ← 训练进度监控
    ├── deploy_farm02.py / deploy_phase2_fix.py  ← 部署脚本
    ├── kill_all.py                     ← 批量终止训练
    ├── scan_all_farms.py               ← 集群巡检
    └── check_phase1_status.py          ← Phase 1 状态
```

---

## 版本演进路线

```
V1 (2025-Q4)    DenseNet3D-201, 单级全局池化, 4 visual tokens
       │         参数量大，信息压缩过度
       ▼
V2 (2026-Q1)    DenseNet3D 多尺度 + Transformer 跨尺度融合
       │         Spark3D CVPR 2025 启发
       ▼
V3 (2026-Q2)    DenseNet3D 四级金字塔, 96 visual tokens (32×3mod)
       │         显存 >29GB，训练速度瓶颈
       ▼
V4 (2026-Q2)    BrainMVP UniFormer (CNN+Attention混合)
       │         更轻量但仍受限于 Attention 在 3D 上的开销
       ▼
V5 (2026-Q3)    MRRCNN 纯 CNN + SE3D + StageAttention
       │         显存 9GB，速度 0.37s/step，三阶段渐进训练
       ▼
    当前版本
```

### 核心指标对比

| 维度 | V1 (DenseNet) | V3 (Spark3D) | V4 (UniFormer) | **V5 (MRRCNN)** |
|------|:---:|:---:|:---:|:---:|
| 编码器 backbone | DenseNet3D-201 | DenseNet3D-201 ×4stages | UniFormer (CNN+Attn) | MRRCNN (纯 CNN) |
| 参数量 (encoder) | 25M×3 | 25M×3 | 22M×3 | **28.8M×3** |
| Visual tokens | 4 | 96 (32×3mod) | 120 (40×3mod) | **120 (40×3mod)** |
| 训练显存 (batch=8) | ~16GB | ~22GB | 29GB | **9GB** |
| 单步速度 | ~2s | ~3s | 9.7s | **0.37s** |
| 瓶颈 | GPU 计算 | GPU 计算 | GPU 计算 | **NAS I/O** |

---

## V1-V3: DenseNet3D 时代

### V1 — 朴素 DenseNet
- **架构**：DenseNet3D-201 独立编码每模态 → 全局池化 → Linear(2048) → 4 visual tokens
- **训练**：三模态 MAE 分别预训练 → 简单 concat → Qwen LoRA 端到端
- **问题**：单级压缩丢失空间信息，4 tokens 远不足以表达完整影像

### V2 — 多模态合并
- `merge_lightweight.py` 实现三模态训练后合并，轻量投影层桥接 DenseNet 输出到 Qwen embedding 空间

### V3 — Spark3D 多尺度融合
- **架构**：DenseNet3D 四级 stage 输出 → 每级 AdaptiveAvgPool + Conv1×1 → 8 tokens × 4 stages = 32 tokens / 模态
- **融合**：2-layer TransformerBlock Self-Attention 跨尺度建模
- **输出**：96 visual tokens（32×3mod） × 512-dim → Linear(2048)
- **参数量**：106M encoder + 54M trainable

---

## V4: BrainMVP UniFormer

### 架构
- **Backbone**：UniFormer (CNN + Local MHA 混合)，depth=[3,4,8,3]，4 级金字塔
- **每模态**：4 级特征各 10 tokens → 40 tokens × 512-dim
- **输出**：120 visual tokens（40×3mod）

### 优势
- CNN 早期层捕获局部纹理，Attention 后期层建模全局依赖
- 在对比学习预训练中表现优异

### 瓶颈
- 3D 体积展平后 self-attention token 数极大（Stage 3: 64 tokens × MHA）
- 显存 29GB（batch=8），训练速度 9.7s/step
- Qwen 全程参与训练 → OOM 频繁

---

## V5: MRRCNN（当前版本）

### 架构
- **Backbone**：纯 6 级 CNN 金字塔 + SE3D 通道注意力 + 轻量 StageAttention
- **输出**：每模态 40 个 CNN tokens × 512-dim → Linear(2048) → 120 visual tokens
- **DiseaseAwareProjector**（Phase 2）：37 个可学 disease query 做 Cross-Attention，替代 mean-pooling 塌缩

### 三阶段渐进训练

```
Phase 1: 对比学习（无需任何外部模型）
  └── 三模态各自 MRRCNN → 对比 loss（同患者 T1/T2/FLAIR 互为正样本）
  └── 显存 9GB，batch=48 无 OOM

Phase 2: Disease-Aware Projector（仅需 Qwen embedding 查表）
  └── 冻结 MRRCNN → 训练 37-query Cross-Attn + 投影层
  └── 视觉侧与文本侧对称 Cross-Attn → 逆频率加权 MSE
  └── 37 个 disease token 各自独立对齐

Phase 3: Qwen LoRA（冻结 V5 encoder + Projector）
  └── 37 visual tokens 作为前缀注入 Qwen → 自回归生成报告
  └── LoRA r=8，3.7M 可训练参数
```

### 关键改进
1. **纯 CNN 无 Attention 开销**：6 级 Conv→SE→Pool，对比度学习 batch 可翻倍
2. **DiseaseAwareProjector**：37 个可解释疾病维度，逆频率加权防高频倾斜
3. **对称 Cross-Attention**：视觉/文本共用 query，两侧投影到同一语义空间
4. **显存与速度**：9GB ≤ V4 的 1/3，0.37s/step

---

## 资料索引

| 文档 | 位置 | 内容 |
|------|------|------|
| 项目原始说明 | `docs/PROJECT_README.md` | V1 DenseNet + Qwen 架构 |
| V5 架构原理 | `docs/V5_项目架构与技术原理.md` | MRRCNN 技术细节 |
| V5 开发总结 | `docs/V5_项目总结.md` | 训练流程、服务器、状态 |
| Phase 3 根因分析 | `docs/V5_Phase3_根因分析报告.md` | mean-pooling 导致的 token 塌缩 |
| FedAvg 指南 | `docs/FedAvg_MRCNN_操作手册.md` | FedAvg 实验步骤 |

---

## 环境要求

| 组件 | 详情 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.x (CUDA 12.8) |
| GPU | ≥24GB (推荐 RTX PRO 6000) |
| 数据集 | MR-RATE (NAS: `/mnt/nas1/disk07/public/mr_data/MR-RATE`) |
| Qwen | Qwen2.5-3B-Instruct |

---

## 联系人

jiaqigu
