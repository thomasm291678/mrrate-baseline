# MR-RATE 项目评估模块 v2

融合了 江上姝 evaluation.v1 到统一评估脚本中。

## 功能

- **37 类病理标签**（关键词提取 + 否定检测），覆盖脑血管/肿瘤/脊柱/先天性/脑实质/退行性/感染等全类别
- **NLG**: BLEU-4 + METEOR + ROUGE-L
- **Diversity**: 去重率 + 平均长度
- **加权综合分**: Composite = 0.4 × NLG_avg + 0.4 × Clinical_Macro_F1 + 0.2 × Uniqueness
- **逐病理 F1**：每个疾病独立输出 Precision/Recall/F1/Support，F1=0 的类目自动提示
- **版本对比**：`--compare` 横向对比多个模型版本

## Usage

```bash
# 标准评估
python evaluation/v2/evaluation_v2.py --preds preds.json --refs refs.json

# 输出结果到 JSON
python evaluation/v2/evaluation_v2.py --preds preds.json --refs refs.json --output results.json

# 多版本对比
python evaluation/v2/evaluation_v2.py --preds v0.json --refs refs.json \
    --compare v1.json v2.json
```

## 已知局限

用关键词匹配替代 Rabert 做标签提取，存在系统性误差（"rule out infarct" 会被误标、复杂否定可能漏检、隐含诊断无法检测），但误差是系统性的，模型越好分数也越高，不影响跨版本对比的有效性。

## 存储位置

- GitHub: [thomasm291678/mrrate-baseline](https://github.com/thomasm291678/mrrate-baseline) `evaluation/`
- NAS: `/mnt/nas1/disk07/public/jiaqigu/evaluation/`
- 本地: `C:\Users\HP\Documents\5555\evaluation\`

## 来源

| 来源 | 贡献 |
|------|------|
| 江上姝 evaluation.v1 | 37 类病理标签 + 否定检测 + Composite 加权 + 原 calc_scores.py 逐病理 F1 |
| 原有 unified_eval.py | NLG/Diversity 基础逻辑 |

融合后统一为单文件 `evaluation_v2.py`，原 JSS 分散的多文件版本保存在 `reference/evaluation_v1_jss/` 供参考。
