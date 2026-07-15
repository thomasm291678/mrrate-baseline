# MR-RATE DenseNet + Qwen 报告生成项目

## 环境要求

- Python 3.12 + CUDA 12.8
- PyTorch nightly (`2.12.0.dev+cu128`)
- GPU: ≥24GB 显存 (推荐 PRO 6000 / 96GB)
- 数据: MR-RATE 数据集 (NAS: `/mnt/nas1/disk07/public/mr_data/MR-RATE`)

## 项目结构

```
mrrate_hidnet/
├── src/
│   ├── densenet3d.py          # DenseNet3D-201 3D 卷积骨干
│   ├── single_modal_encoder.py # 单模态自编码器 (MAE)
│   ├── mrrate_dataset.py       # MR-RATE 真实数据加载器
│   ├── model.py                # 原始多模态模型 (备用)
│   ├── hidnet.py               # 原始 HIDNet (备用)
│   └── ...                     # 其他工具
├── scripts/
│   ├── pretrain_densenet.py    # ★ Stage 1: DenseNet MAE 预训练
│   ├── merge_lightweight.py    # ★ Stage 2: 合并 3 模态 encoder
│   ├── train_report_gen.py     # ★ Stage 3: 投影对齐 + Qwen LoRA
│   └── ...
├── configs/
│   └── baseline.yaml           # 原始训练配置
├── outputs/
│   ├── pretrain_densenet/      # Stage 1 输出
│   │   ├── best_t1.pt          # T1 预训练权重 (1.1GB)
│   │   ├── best_t2.pt          # T2 预训练权重
│   │   ├── best_flair.pt       # Flair 预训练权重
│   │   └── multimodal_encoder.pt  # Stage 2 合并输出
│   └── report_gen/             # Stage 3 输出
│       └── best_model.pt       # 最终模型 (9.7GB)
└── requirements.txt
```

## 模型架构

### 数据流

```
MRI 扫描 (T1 + Flair + T2, 各 128³)
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 1: DenseNet3D-201 (冻结)     │
│  每个模态独立的 MAE 预训练           │
│  25M 参数/模态 × 3 = 75M            │
│  输出: (B, 2048) 特征向量            │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 2: 投影层 (可训练)            │
│  Linear(2048 → 2048×4)              │
│  输出: (B, 4, 2048) visual tokens    │
│  3模态 concat → (B, 12, 2048)       │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Stage 3: Qwen2.5-3B (LoRA)        │
│  LoRA: r=8, attention_q/k/v/o       │
│  输入: 12 visual tokens + text       │
│  输出: 放射学报告文本                │
└─────────────────────────────────────┘
```

### 参数统计

| 组件 | 参数 | 可训练 | 说明 |
|------|------|--------|------|
| DenseNet ×3 | 75M | 0 (冻结) | Stage 1 预训练 |
| 投影层 ×3 | 50M | 50M | Stage 2 初始化 |
| Qwen LoRA | 3.7M | 3.7M | Stage 3 微调 |
| **合计** | **882M** | **54M** | |

## 训练流程

### Stage 1: DenseNet MAE 预训练 (本地 5060)

```bash
# 三模态分别训练
python scripts/pretrain_densenet.py --modality t1   --batch_size 4 --use_amp --epochs 100
python scripts/pretrain_densenet.py --modality t2   --batch_size 4 --use_amp --epochs 100
python scripts/pretrain_densenet.py --modality flair --batch_size 4 --use_amp --epochs 100

# 中断续训
python scripts/pretrain_densenet.py --modality flair --batch_size 4 --use_amp \
  --resume outputs/pretrain_densenet/best_flair.pt
```

- 输入: 单模态 3D 体积 (128³), 随机 mask 75%
- 任务: MAE 重建 (MSE loss)
- 显存: ~7.4GB, 56s/epoch (32 合成样本)
- 最终 MSE: ~0.00018 (基线 MSE=0.0045, 改善 25x)

### Stage 2: 合并编码器 (任意)

```bash
python scripts/merge_lightweight.py \
  --t1 outputs/pretrain_densenet/best_t1.pt \
  --t2 outputs/pretrain_densenet/best_t2.pt \
  --flair outputs/pretrain_densenet/best_flair.pt
```

- 输出: `outputs/pretrain_densenet/multimodal_encoder.pt`
- 3 个 encoder (冻结) + 3 个投影头 (随机初始化, 需 Stage 3 训练)

### Stage 3: Qwen 对齐训练 (farm05)

```bash
nohup python -u scripts/train_report_gen.py \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --encoder_ckpt outputs/pretrain_densenet/multimodal_encoder.pt \
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
  --batch_size 2 --epochs 5 --lr 2e-5 --use_amp \
  --resume outputs/report_gen/best_model.pt \
  > outputs/report_gen/train.log 2>&1 &
```

- 数据: MR-RATE 88k 训练样本
- 任务: 生成放射学报告 (CE loss)
- 训练: 投影层 + Qwen LoRA (54M 可训练参数)
- 显存: ~16GB (batch=2)

## 数据格式

### MR-RATE 数据集

```
/mnt/nas1/disk07/public/mr_data/MR-RATE/
├── splits.csv          # batch_id, patient_uid, study_uid, split
├── mri/{batch}/{study}/img/{study}_{modality}.nii.gz
│   ├── t1w-raw-sag.nii.gz
│   ├── flair-raw-sag.nii.gz
│   └── t2w-raw-axi.nii.gz
└── reports/{batch}_reports.csv  # study_uid, report, findings, ...
```

### 数据加载器

使用 `src/mrrate_dataset.py`:
```python
from src.mrrate_dataset import MRRateDataset
ds = MRRateDataset("/path/to/MR-RATE", "train")
sample = ds[0]  # {t1, flair, t2, has_t1, has_flair, has_t2, report, study_uid}
```

## 推理

```python
from scripts.merge_lightweight import LightweightMultiModal
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 加载模型
encoder = LightweightMultiModal().cuda()
encoder.load_state_dict(torch.load('outputs/report_gen/best_model.pt')['encoder_state'])

llm = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
llm = get_peft_model(llm, LoraConfig(r=8, ...))
llm.load_state_dict(ckpt['llm_state'])

# 编码 MRI → visual tokens
vtok = encoder(t1, flair, t2, has_t1, has_flair, has_t2)  # (1, 12, 2048)

# 注入 Qwen 生成报告
embeds = torch.cat([vtok, text_embeddings], dim=1)
output = llm.generate(inputs_embeds=embeds, ...)
```

## 关键技术决策

1. **为什么分开训练而不是端到端？**
   - 8GB 显存放不下 DenseNet + Qwen 同时训练
   - 分阶段训练: 先 MAE 预训练 (本地 5060), 再对齐 Qwen (服务器)

2. **为什么不用跨模态融合？**
   - 用户要求: "把三个维度连起来那部分删掉"
   - 单模态独立训练 + 输出时简单 concat
   - 后面的聚合不需要神经网络

3. **投影层为什么需要训练？**
   - DenseNet 输出 → 投影层映射到 Qwen embedding 空间
   - 随机初始化的投影层 = Qwen 收到随机噪声
   - Stage 3 的训练就是学习这个映射

4. **为什么用 LoRA 而不是全量微调？**
   - Qwen 3B 全量 ~6GB, LoRA 只有 3.7M
   - 保留 Qwen 的语言能力, 只适配医学报告风格

## 复现步骤

```bash
# 1. 环境
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --pre
pip install transformers peft nibabel scipy pandas pyyaml tqdm

# 2. Stage 1: MAE 预训练 (本地, 各 ~1.5h)
python scripts/pretrain_densenet.py --modality t1 --batch_size 4 --use_amp --epochs 100
python scripts/pretrain_densenet.py --modality t2 --batch_size 4 --use_amp --epochs 100
python scripts/pretrain_densenet.py --modality flair --batch_size 4 --use_amp --epochs 100

# 3. Stage 2: 合并
python scripts/merge_lightweight.py \
  --t1 outputs/pretrain_densenet/best_t1.pt \
  --t2 outputs/pretrain_densenet/best_t2.pt \
  --flair outputs/pretrain_densenet/best_flair.pt

# 4. Stage 3: 对齐训练 (服务器, ~数小时)
python scripts/train_report_gen.py \
  --data_root /path/to/MR-RATE \
  --qwen_path /path/to/Qwen2.5-3B-Instruct \
  --batch_size 2 --epochs 5 --lr 2e-5 --use_amp

# 5. 推理
python -c "..." # 见上方推理代码
```

## 已知问题

- Stage 3 训练时 Qwen 从 NAS 加载很慢 (NFS folio_wait)
- 建议把模型权重拷贝到本地 SSD
- 88k 全量训练耗时长 (~2s/step × 110k steps ≈ 60h)
