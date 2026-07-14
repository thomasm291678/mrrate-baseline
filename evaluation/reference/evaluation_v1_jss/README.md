# MRI-version — VLM3D MR Report Generation 本地评估工具包

本目录是 VLM3D-MR-Report Generation 挑战赛的**最终本地评估成果**，可独立运行，不依赖官方平台。核心设计目标：用最小成本（无需训模型、无需 GPU）近似模拟官方打分逻辑，用于训练过程中的日常监控和 checkpoint 择优。

## 目录结构

```
MRI-version/
├── eval_local.py               # ★ 主评估入口：端到端综合打分
├── extract_labels_keyword.py    # 关键词 label 提取器（代替 RadBERT）
├── calc_scores.py              # CSV 级多标签分类指标（per-label P/R/F1/Acc）
├── crg_score.py                # CRG 临床加权相关性分（含 TP/FN/FP 分解）
├── test_nlg.py                 # NLG 指标冒烟测试（BLEU/METEOR/ROUGE-L）
├── test_clinical_f1.py         # Clinical F1 独立测试（micro + macro）
├── test_diversity.py           # 多样性去重率独立测试
├── preds.json                  # 示例：模型预测报告（100 条）
├── refs.json                   # 示例：参考报告/真值（100 条）
├── pred_labels.json            # 示例：从 preds 提取的 37 维标签
├── ref_labels.json             # 示例：真值 37 维标签
└── results.json                # 示例：最近一次评估输出
```

## 架构总览

```
                           ┌──────────────────────────┐
                           │   extract_labels_keyword  │
                           │  (关键词 + 否定检测)       │
                           │  输入: preds.json         │
                           │  输出: pred_labels.json   │
                           └──────────┬───────────────┘
                                      │
  ┌─────────────┐           ┌─────────▼──────────┐
  │  preds.json  │──────────▶│    eval_local.py    │
  │  refs.json   │           │   (加权综合打分)     │────▶ 终端输出 + results.json
  └─────────────┘           └────────────────────┘
                                      ▲
  ┌─────────────────┐                 │
  │ ref_labels.json  │────────────────┘
  └─────────────────┘
                                      ▲
  ┌──────────────────┐                │
  │  calc_scores.py   │───────────────┘ (CSV → per-label 指标)
  │  crg_score.py     │───────────────┘ (CSV → CRG 分)
  └──────────────────┘
```

## 核心文件说明

### `eval_local.py` — 主评估入口 ★

端到端加权综合打分脚本，一次运行输出所有指标。

**使用方式：**
```bash
# 仅 NLG + 多样性（无 label 文件时）
python eval_local.py --preds ./preds.json --refs ./refs.json

# 完整评估：NLG + Clinical F1 + 多样性 + 综合分
python eval_local.py \
    --preds ./preds.json \
    --refs ./refs.json \
    --pred-labels ./pred_labels.json \
    --labels ./ref_labels.json \
    --output results.json
```

**输入格式：**

`preds.json` / `refs.json`：
```json
[
  {
    "report_id": "exam_001",
    "report": "MRI brain shows no acute intracranial hemorrhage..."
  },
  ...
]
```

`pred_labels.json` / `ref_labels.json`：
```json
[
  {
    "Cerebral infarction": 0,
    "Cerebral hemorrhage": 0,
    "Gliosis": 1,
    ...（共 37 个 key，每个 value 为 0 或 1）
  },
  ...
]
```

**打分公式：**
```
综合分 = 0.40 × NLG_avg  +  0.40 × Clinical_Macro_F1  +  0.20 × Diversity

其中:
  NLG_avg        = (BLEU-4 + METEOR + ROUGE-L) / 3
  Clinical_Macro_F1 = sklearn f1_score(average='macro', zero_division=0)
  Diversity      = unique_reports / total_reports（归一化去标点空格后统计）
```

**输出：**
- 终端打印所有指标分组
- 可选 `--output results.json` 保存结构化 JSON

---

### `extract_labels_keyword.py` — 无模型 Label 提取器

从生成报告中提取 37 类二值疾病标签，**完全基于关键词匹配 + 多层否定检测**，不依赖任何模型（不需要 GPU、不需要下载 RadBERT）。

**为什么需要它：** 官方平台用 RadBERT（37 类分类器）从报告中提取标签再计算 Clinical F1。本地没有 RadBERT，此脚本用规则引擎近似之，5 秒跑完 100 条。

**使用方式：**
```bash
python extract_labels_keyword.py preds.json -o pred_labels.json
```

**37 类标签覆盖（按类别）：**

| 类别 | 数量 | 示例 |
|------|------|------|
| 脑血管疾病 | 7 | Cerebral infarction, Cerebral hemorrhage, Lacunar infarct, ... |
| 肿瘤 | 6 | Glioma, Meningioma, Schwannoma, Pituitary adenoma, ... |
| 脊柱疾病 | 5 | Herniation, Spinal stenosis, Spinal cord compression, ... |
| 先天性/良性变异 | 5 | Arachnoid cyst, Pineal cyst, Rathke's pouch cyst, ... |
| 脑实质病变 | 3 | Gliosis, Encephalomalacia, Cerebral edema |
| 退行性病变 | 2 | Cerebral atrophy, Cerebellar degeneration |
| 其他 | 9 | Ventriculomegaly, Mastoiditis, Chiari malformation, ... |

**否定检测（三层防御）：**
1. **直接前置否定词** — 匹配位置前 3 个词内有 `no/not/without/negative` 则丢弃
2. **特定否定句式** — 前后 80 字符内出现 `"was not detected"`, `"no evidence of"`, `"no area of restricted diffusion"`, `"ruled out"` 等 10 种放射报告常见否定句式则丢弃
3. **简短前置否定** — `"no acute infarct"` 型直接放弃匹配

**已验证性能（100 条 ground truth 报告）：**
- Micro F1: 0.82
- Macro F1: 0.42（受限于 14/37 标签无正样本 + 罕见标签漏检）
- 高频标签表现优秀：Gliosis 0.92, Ventriculomegaly 0.92, Cerebral atrophy 0.77

> ⚠️ **重要认知：** 绝对分数值和官方有差距，但**不同 checkpoint 之间的趋势判断是正确的**——而这正是本地评估唯一需要的。

---

### `calc_scores.py` — CSV 级多标签分类指标

从 CSV 文件计算 per-label precision/recall/f1/accuracy。

**使用方式：**
```bash
python calc_scores.py --pred_csv preds.csv --gt_csv gt.csv --out_json classification_scores.json
```

**输入格式：** CSV 文件，第一列为 `AccessionNo`，其余 37 列为 0/1 标签。

**输出格式：**
```json
{
  "per_pathology": [
    {"name": "Cerebral infarction", "precision": 0.88, "recall": 0.79, "f1": 0.83, "accuracy": 0.91},
    ...
  ],
  "macro": {"precision": 0.85, "recall": 0.82, "f1": 0.83, "accuracy": 0.90}
}
```

---

### `crg_score.py` — 临床加权相关性分（CRG）

CRG (Clinically-Weighted Relevance) 是专门针对多标签医学分类设计的综合指标，通过类不平衡比率 `r` 对 TP/FN/FP 赋予不同权重。

**公式：**
```
X  = 37 × N          （总标签位数量）
A  = GT 中阳性标签总数
r  = (X - A) / (2A)  （不平衡权重因子）
U  = (X - A) / 2
s  = r·TP - r·FN - FP
CRG = U / (2U - s)
```

**使用方式：**
```bash
python crg_score.py --pred_csv preds.csv --gt_csv gt.csv --out_json crg.json
```

---

### 测试脚本

| 文件 | 用途 | 用法 |
|------|------|------|
| `test_nlg.py` | 验证 BLEU-4 / METEOR / ROUGE-L 计算正常 | `python test_nlg.py` |
| `test_clinical_f1.py` | 验证 Clinical F1 (micro+macro) 计算正常 | `python test_clinical_f1.py pred_labels.json ref_labels.json` |
| `test_diversity.py` | 验证多样性去重率计算正常 | `python test_diversity.py preds.json` |

---

## 典型工作流

### 日常训练监控（推荐）

```bash
# Step 1: 模型推理，生成报告
python your_model/inference.py --checkpoint epoch_10.pt --output preds.json

# Step 2: 从报告中提取 37 类标签（5 秒）
python MRI-version/extract_labels_keyword.py preds.json -o pred_labels.json

# Step 3: 完整评估
python MRI-version/eval_local.py \
    --preds preds.json \
    --refs MRI-version/refs.json \
    --pred-labels pred_labels.json \
    --labels MRI-version/ref_labels.json \
    --output epoch_10_scores.json
```

### 关键 checkpoint 提交前（可选增强）

日常用关键词匹配；提交前可换 LLM API 做更准的 label 提取（如 Qwen/GPT），提取流程不变，只替换 Step 2 的实现。

### 快速冒烟

```bash
# 验证所有指标计算正常工作
python MRI-version/test_nlg.py
python MRI-version/test_clinical_f1.py MRI-version/pred_labels.json MRI-version/ref_labels.json
python MRI-version/test_diversity.py MRI-version/preds.json
```

## 依赖

```bash
pip install nltk rouge-score scikit-learn numpy pandas
```

首次使用需下载 NLTK 数据：
```bash
python -c "import nltk; nltk.download('wordnet')"
```

## 设计原则

1. **无模型依赖** — label 提取不用任何神经网络，纯规则引擎
2. **单文件即用** — 每个脚本独立可运行，无内部 import 依赖
3. **趋势正确 > 绝对值精准** — 设计目标是区分 checkpoint 好坏，不是完美复现官方分数
4. **40/40/20 权重** — 与队友方案对齐，三个维度同等覆盖

## 已知局限

- **Macro F1 被零样本标签拖累**：数据集中 14/37 类无正样本，`zero_division=0` 下 F1=0，拉低 macro 均值。这是数据分布决定的，不是脚本 bug
- **罕见标签漏检**：只有 1-2 个正样本的标签（如 Lipoma of brain、Hyperostosis of skull）关键词匹配难以命中
- **否定检测不完美**：复杂嵌套否定句（"the possibility of infarction cannot be excluded"）仍可能误判
- **去重率粗糙**：仅做标点/空格归一化后比较，不含语义去重
