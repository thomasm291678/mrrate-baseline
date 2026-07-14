#!/usr/bin/env python3
"""最简 NLG 测试 —— 验证 BLEU/METEOR/ROUGE-L 能跑通"""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

ref = "Brain MRI shows no acute intracranial hemorrhage or mass effect."
hyp = "MRI brain demonstrates no acute hemorrhage or mass effect."

# BLEU-4
smooth = SmoothingFunction().method1
bleu = sentence_bleu([ref.split()], hyp.split(),
                     weights=(0.25, 0.25, 0.25, 0.25),
                     smoothing_function=smooth)
print(f"BLEU-4:  {bleu:.4f}")

# METEOR
meteor = meteor_score([ref.split()], hyp.split())
print(f"METEOR:  {meteor:.4f}")

# ROUGE-L
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge = scorer.score(ref, hyp)['rougeL'].fmeasure
print(f"ROUGE-L: {rouge:.4f}")
