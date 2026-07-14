#!/usr/bin/env python3
"""Clinical F1 测试 —— 计算 micro + macro F1，支持真实数据文件"""
import json
import sys
import numpy as np
from sklearn.metrics import f1_score

def compute_f1(pred_labels, ref_labels):
    """pred_labels/ref_labels: list[dict], 每个 dict 包含 37 类 0/1"""
    label_names = list(ref_labels[0].keys())
    y_true = np.array([[lb[name] for name in label_names] for lb in ref_labels])
    y_pred = np.array([[lb[name] for name in label_names] for lb in pred_labels])

    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return macro, micro

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        # 从文件读入
        with open(sys.argv[1]) as f:
            pred_labels = json.load(f)
        with open(sys.argv[2]) as f:
            ref_labels = json.load(f)
        print(f"加载 {len(pred_labels)} 条样本")
    else:
        # 模拟数据（37 类，第 3 个标签预测错）
        print("未指定文件，使用模拟数据:\n")
        y_true = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred_labels = [dict(zip([f"label_{i}" for i in range(37)], y_pred))]
        ref_labels  = [dict(zip([f"label_{i}" for i in range(37)], y_true))]

    macro, micro = compute_f1(pred_labels, ref_labels)
    print(f"Micro F1:  {micro:.4f}")
    print(f"Macro F1:  {macro:.4f}")

    # 附加信息
    label_names = list(ref_labels[0].keys())
    ever_pos = sum(1 for n in label_names if any(s[n] for s in ref_labels))
    print(f"标签覆盖率: {ever_pos}/{len(label_names)} ({100*ever_pos/len(label_names):.0f}%)")
    if ever_pos < len(label_names):
        print("⚠️ 部分标签正样本=0，macro F1 会被 zero_division=0 拉低")
