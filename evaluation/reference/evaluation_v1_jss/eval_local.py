#!/usr/bin/env python3
"""
端到端本地评估脚本（整合队友方案：加权综合分）
用法: python eval_local.py --preds ./my_preds.json --refs ./refs.json [--pred-labels ./my_labels.json] [--labels ./ref_labels.json]
"""
import json
import argparse
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

# ============================================================
# 权重配置（队友方案 40/40/20，可调）
# ============================================================
WEIGHT_NLG       = 0.40   # BLEU-4 + METEOR + ROUGE-L 均值
WEIGHT_CLINICAL  = 0.40   # Clinical Macro F1（37 类）
WEIGHT_DIVERSITY = 0.20   # 多样性去重率

# ============================================================
# 1. NLG 指标
# ============================================================
def compute_nlg(pred_reports, ref_reports):
    """计算 BLEU-4, METEOR, ROUGE-L，返回 case-level 列表和均值"""
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_list, meteor_list, rouge_list = [], [], []
    for pred, ref in zip(pred_reports, ref_reports):
        pred_tok = pred.split()
        ref_tok = ref.split()
        bleu_list.append(sentence_bleu(
            [ref_tok], pred_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth))
        meteor_list.append(meteor_score([ref_tok], pred_tok))
        rouge_list.append(scorer.score(ref, pred)['rougeL'].fmeasure)

    return {
        'BLEU-4':  float(np.mean(bleu_list)),
        'METEOR':  float(np.mean(meteor_list)),
        'ROUGE-L': float(np.mean(rouge_list)),
    }

# ============================================================
# 2. Clinical Label F1（37 类，来自 MR-RATE §6.3）
# ============================================================
def compute_clinical_f1(pred_labels, ref_labels):
    """
    pred_labels / ref_labels: list of dict, 每个 dict 是 37 维 0/1 标签
    返回 micro + macro F1
    """
    label_names = list(ref_labels[0].keys())
    y_true = np.array([[lb[name] for name in label_names] for lb in ref_labels])
    y_pred = np.array([[lb[name] for name in label_names] for lb in pred_labels])

    return {
        'Clinical_Macro_F1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'Clinical_Micro_F1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
    }

# ============================================================
# 3. 多样性去重率（队友方案 20% 权重）
# ============================================================
def compute_diversity(pred_reports):
    """
    报告级去重率：规范化后统计唯一报告占比。
    1.0 = 每条报告都不同（完全多样）
    趋近 0.0 = 所有报告完全相同（mode collapse）
    """
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    normed = [normalize(r) for r in pred_reports]
    unique = len(set(normed))
    return unique / len(normed) if normed else 0.0

# ============================================================
# 4. 加权综合分
# ============================================================
def compute_composite(nlg_scores, clinical_scores, diversity_score):
    """加权综合分 = 40% NLG + 40% Clinical + 20% Diversity"""
    nlg_avg = (nlg_scores['BLEU-4'] + nlg_scores['METEOR'] + nlg_scores['ROUGE-L']) / 3.0
    clinical_avg = clinical_scores['Clinical_Macro_F1']  # 默认用 macro
    return (WEIGHT_NLG * nlg_avg
            + WEIGHT_CLINICAL * clinical_avg
            + WEIGHT_DIVERSITY * diversity_score)

# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='预测报告 JSON')
    ap.add_argument('--refs', required=True, help='参考报告 JSON')
    ap.add_argument('--pred-labels', default=None, help='预测标签 JSON (37类, 可选)')
    ap.add_argument('--labels', default=None, help='参考标签 JSON (37类, 可选)')
    ap.add_argument('--output', default=None, help='输出 JSON 路径 (可选)')
    args = ap.parse_args()

    # 加载数据
    with open(args.preds) as f:
        preds = json.load(f)
    with open(args.refs) as f:
        refs = json.load(f)

    pred_texts = [p['report'] for p in preds]
    ref_texts  = [r['report'] for r in refs]

    # --- NLG ---
    nlg = compute_nlg(pred_texts, ref_texts)

    # --- Clinical F1 ---
    clinical = {}
    if args.labels and args.pred_labels:
        with open(args.pred_labels) as f:
            pred_labels = json.load(f)
        with open(args.labels) as f:
            ref_labels = json.load(f)
        clinical = compute_clinical_f1(pred_labels, ref_labels)

    # --- Diversity ---
    diversity = compute_diversity(pred_texts)

    # --- Composite ---
    composite = None
    if clinical:
        composite = compute_composite(nlg, clinical, diversity)

    # --- 输出 ---
    print("=" * 50)
    print("  NLG Metrics (40%)")
    print("=" * 50)
    for k, v in nlg.items():
        print(f"  {k:20s}: {v:.4f}")

    if clinical:
        print("\n" + "=" * 50)
        print("  Clinical Label F1 (40%) — 37 classes")
        print("=" * 50)
        for k, v in clinical.items():
            print(f"  {k:20s}: {v:.4f}")

    print("\n" + "=" * 50)
    print("  Diversity (20%)")
    print("=" * 50)
    print(f"  多样性去重率        : {diversity:.4f}")

    if composite is not None:
        print("\n" + "=" * 50)
        print("  Weighted Composite Score")
        print("=" * 50)
        print(f"  Composite (40/40/20): {composite:.4f}")

    print(f"\n✅ 评估完成")

    # 可选：保存到文件
    if args.output:
        result = {
            'nlg': nlg,
            'clinical': clinical,
            'diversity': diversity,
            'composite': composite,
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"📁 结果已保存到 {args.output}")

if __name__ == '__main__':
    main()