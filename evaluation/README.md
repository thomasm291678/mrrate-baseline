# MR-RATE 项目评估模块 v2

## 版本演进

| Version | 文件 | 说明 |
|---------|------|------|
| v1 (legacy) | `unified_eval.py` | 原始版本，14 类 Clinical 标签 + NLG + Diversity |
| evaluation.v1 (JSS) | `reference/evaluation_v1_jss/` | 江上姝基于 VLM3D-Dockers 改编，37 类病理标签 + 否定检测 + 加权综合分 |
| **v2 (current)** | **`evaluation_v2.py`** | **融合版，包含以下全部能力** |

## evaluation_v2.py 特性

### 融合内容

1. **37 类病理标签** (from JSS evaluation.v1)
   - 关键词提取 + 逐词否定检测窗口
   - 覆盖脑血管/肿瘤/脊柱/先天性/脑实质/退行性/感染等全类别
   - 输出 Macro F1 + Micro F1 + 逐病理 Precision/Recall/F1/Accuracy/Support

2. **14 类简化标签** (from unified_eval.py)
   - 作为 legacy 参考保留，与 37 类并行输出

3. **NLG Metrics**
   - BLEU-4 (nltk smoothing function)
   - METEOR
   - ROUGE-L

4. **Diversity**
   - 去重率 (uniqueness/duplicate_ratio)
   - 平均长度 + Most common dup

5. **加权综合分**
   - `Composite = 0.4 * NLG_avg + 0.4 * Clinical_Macro_F1 + 0.2 * Uniqueness`
   - 权重可在代码顶部 `WEIGHT_*` 调整

6. **版本对比**
   - `--compare v1_preds.json v2_preds.json` 横向对比多轮结果
   - 自动标记最高分

### Usage

```bash
# 标准评估
python evaluation_v2.py --preds preds.json --refs refs.json

# 输出结果到 JSON
python evaluation_v2.py --preds preds.json --refs refs.json --output results.json

# 带预提取的 ground truth labels
python evaluation_v2.py --preds preds.json --refs refs.json \
    --gt-labels gt_labels_37.json --output results.json

# 仅 NLG + Diversity（无 Clinical）
python evaluation_v2.py --preds preds.json --refs refs.json --no-clinical

# 多版本对比
python evaluation_v2.py --preds v0_preds.json --refs refs.json \
    --compare v1_preds.json v2_preds.json
```

### 已知局限

**关键词提取（替代 Rabert 的 trade-off）：**
- "提及但未确诊"的疾病会被误标为阳性（如 "rule out infarction" → infarction=1）
- 否定检测覆盖常见否定句式，但复杂否定结构可能漏掉
- 无法检测隐含诊断（如 "mass effect with surrounding edema suggesting neoplasm"）

**但误差是系统性的**：模型变好 → 关键词匹配分数也变好，趋势一致，不影响评估的有效性。

## 文件结构

```
evaluation/
├── evaluation_v2.py              # 融合版主脚本（当前使用）
├── v1_legacy/
│   └── unified_eval.py           # 原始 14 类版本
└── reference/
    └── evaluation_v1_jss/        # 江上姝原版（供参考）
        ├── eval_local.py         # 主评估入口
        ├── extract_labels_keyword.py  # 37 类标签提取
        ├── calc_scores.py        # 逐病理 F1 计算
        ├── crg_score.py          # CRG（平台侧用，已排除）
        └── test_*.py             # 各模块单元测试
```

## 存储位置

- **本地**: `C:\Users\HP\Documents\5555\evaluation\`
- **NAS**: `/mnt/nas1/disk07/public/jiaqigu/evaluation/`
- **GitHub**: `github.com/jiaqigu/mrrate-hidnet/evaluation/`

## 与 JSS evaluation.v1 的关系

### 改进点

1. **单文件聚合**：evaluation.v1 分 `eval_local.py` + `extract_labels_keyword.py` + `calc_scores.py` 三个文件，v2 合并为单文件 `evaluation_v2.py`，减少交叉依赖
2. **14 类并行保留**：legacy 14 类标签与 37 类标签在同一轮评估中输出，方便对比
3. **逐病理 F1 整合到主流程**：evaluation.v1 里 `calc_scores.py` 未接入主 `eval_local.py`，v2 直接在 evaluate() 中输出
4. **难识别疾病提示**：自动标记 F1=0 的疾病类别，方便针对性加 trick
5. **版本对比增强**：evaluation.v1 无版本对比功能，v2 的 `compare_versions()` 支持横向对比

### 效果预期

- **37 类 Clinical F1** 预计比 14 类略低（标签更细粒度，更难匹配）
- **但诊断覆盖度显著提升**：从 14 个大类→37 个具体病理，能更精细地评估模型对罕见/特定疾病的表现
- **逐病理 F1** 可以直接看到哪些疾病是模型短板，指导后续数据增强/trick 方向
- **Composite 分数** 提供单一数字方便对比版本间的综合提升
