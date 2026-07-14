#!/usr/bin/env python3
"""多样性去重率测试 —— 检测生成报告是否模板化，支持真实数据文件"""
import re
import json
import sys

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 去标点
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def diversity_dedup_rate(reports):
    """
    报告级去重率。1.0 = 每条都不同，趋近 0.0 = mode collapse。
    """
    normed = [normalize(r) for r in reports]
    unique = len(set(normed))
    return unique / len(normed) if normed else 0.0

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            preds = json.load(f)
        reports = [p['report'] for p in preds]
        print(f"加载 {len(reports)} 条报告")
    else:
        # 模拟数据：4 条中 1 条重复 → 3/4 = 0.75
        print("未指定文件，使用模拟数据:\n")
        reports = [
            "Brain MRI shows no acute intracranial hemorrhage.",
            "Mild cerebral atrophy noted. No mass effect.",
            "Brain MRI shows no acute intracranial hemorrhage.",  # 重复
            "Diffuse white matter disease with ventriculomegaly.",
        ]
        for i, r in enumerate(reports):
            print(f"  [{i+1}] {r}")

    rate = diversity_dedup_rate(reports)
    unique = len(set(normalize(r) for r in reports))
    print(f"\n总报告数:   {len(reports)}")
    print(f"唯一报告:   {unique}")
    print(f"多样性去重率: {rate:.4f}")
