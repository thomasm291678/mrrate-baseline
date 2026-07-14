#!/usr/bin/env python3
"""
MR-RATE Unified Evaluation v2
================================
融合:
  - 江上姝 evaluation.v1: 37 类病理标签（关键词提取 + 否定检测）+ 加权综合分
  - 原始 unified_eval.py: 14 类简化标签 + NLG + Diversity + 版本对比
  - calc_scores.py: 逐病理 F1 输出（保留 37 类各自 Precision/Recall/F1/Accuracy）

Usage:
  # 完整评估（37 类 Clinical + 14 类 Clinical + NLG + Diversity + 综合分）
  python evaluation_v2.py --preds preds.json --refs refs.json

  # 带 ground truth label 文件（用于精确 Clinical F1）
  python evaluation_v2.py --preds preds.json --refs refs.json \\
      --gt-labels gt_labels.json --output results.json

  # 仅 NLG + Diversity（无 label 时）
  python evaluation_v2.py --preds preds.json --refs refs.json --no-clinical

  # 版本对比
  python evaluation_v2.py --preds v1_preds.json --refs refs.json \\
      --compare v2_preds.json v3_preds.json

Author: jiaqigu / shushangjiang
Version: v2.0 (2026-07-14)
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

# ============================================================================
# 权重配置
# ============================================================================
WEIGHT_NLG = 0.40
WEIGHT_CLINICAL = 0.40
WEIGHT_DIVERSITY = 0.20


# ============================================================================
# 37 类病理标签（来自 MR-RATE §6.3，江上姝 evaluation.v1）
# ============================================================================
LABEL_37_PATTERNS = {
    # 脑血管疾病 (7)
    "Cerebral infarction": [
        "cerebral infarct", "brain infarct", "ischemic stroke",
        "acute infarct", "established infarct", "territorial infarct",
    ],
    "Cerebral hemorrhage": [
        r"cerebral h(?:a|e)mor+hage", r"intracerebral h(?:a|e)mor+hage",
        r"brain h(?:a|e)mor+hage", r"intraparenchymal h(?:a|e)mor+hage",
    ],
    "Lacunar infarct": [
        "lacunar infarct", "lacune", "lacunar stroke",
        "cystic lacunar", "chronic lacunar",
    ],
    "Silent micro-hemorrhage of brain": [
        r"microh(?:a|e)mor+hage", "microbleed", "cerebral microbleed",
        r"petechial h(?:a|e)mor+hage", "microbleeds",
    ],
    "Subdural intracranial hemorrhage": [
        r"subdural h(?:a|e)mor+hage", r"subdural h(?:a|e)matoma", "sdh",
    ],
    "Intracranial aneurysm": [
        "aneurysm", "berry aneurysm", "saccular aneurysm",
    ],
    "Watershed infarct": [
        "watershed infarct", "border zone infarct", "borderzone infarct",
        "watershed ischemia",
    ],

    # 肿瘤 (6)
    "Metastatic malignant neoplasm to brain": [
        r"metasta(?:sis|ses|tic)", "brain metasta", r"intracranial metasta",
    ],
    "Intracranial meningioma": ["meningioma"],
    "Schwannoma": [
        "schwannoma", "vestibular schwannoma", "acoustic neuroma", "neurinoma",
    ],
    "Glioma": [
        "glioma", "glioblastoma", "astrocytoma", "oligodendroglioma",
        "ependymoma", "glial tumor", "glial neoplasm",
    ],
    "Pituitary adenoma": [
        "pituitary adenoma", "pituitary macroadenoma", "pituitary microadenoma",
        "sellar mass", "pituitary mass", "pituitary tumor",
    ],
    "Lipoma of brain": ["brain lipoma", "intracranial lipoma"],

    # 脊柱疾病 (5)
    "Herniation of nucleus pulposus": [
        "herniation", "herniated disc", "disc herniation", "herniated disk",
        "disc protrusion", "disc extrusion", "disc bulge",
    ],
    "Spinal cord compression": [
        "cord compression", "spinal cord compression", "thecal sac compression",
    ],
    "Hemangioma of vertebral column": [
        r"vertebral (?:h(?:a|e)mangioma|hemangioma)", r"vertebral body h(?:a|e)mangioma",
    ],
    "Spinal stenosis": [
        "spinal stenosis", "canal stenosis", "central canal stenosis",
        "central stenosis",
    ],
    "Foraminal Spinal Stenosis": [
        "foraminal stenosis", "neural foraminal narrowing", "foraminal narrowing",
        "foraminal encroachment",
    ],

    # 先天性/良性变异 (5)
    "Arachnoid cyst": ["arachnoid cyst"],
    "Cyst of pineal gland": ["pineal cyst", "pineal gland cyst"],
    "Rathke's pouch cyst": [
        r"rathke(?:'s|) (?:pouch |cleft |)cyst", "rathke cleft cyst",
    ],
    "Mega cisterna magna": [
        "mega cisterna magna", "prominent cisterna magna",
        "enlarged cisterna magna",
    ],
    "Structure of cave of septum pellucidum": [
        "cavum septum pellucidum", "cave of septum pellucidum",
        r"\bcsp\b", "cavum vergae",
    ],

    # 脑实质病变 (3)
    "Gliosis": [
        "gliotic", "gliosis", "glial scar",
        "small vessel ischemic disease", "small vessel disease",
        "chronic ischemic changes?", r"white matter (?:ischemic |hyperintens|disease|changes?)",
        "periventricular hyperintens", "leukoaraiosis",
        "microangiopath(?:ic|y)", r"subcortical (?:hyperintens|white matter)",
        "chronic microvascular", "ischemic white matter",
    ],
    "Encephalomalacia": [
        "encephalomalacia", "encephalomalacic",
        "cystic encephalomalaci", "porencephal",
    ],
    "Cerebral edema": [
        "cerebral edema", "brain edema", "cerebral oedema",
        r"vasogenic edema", r"cytotoxic edema", r"parenchymal edema",
        r"peritumoral edema", "brain swelling",
    ],

    # 退行性病变 (2)
    "Cerebral atrophy": [
        "cerebral atrophy", "brain atrophy", "cortical atrophy",
        "cerebral volume loss", "central (?:cerebral |)atrophy",
        "ex vacuo dilatation", "sulcal (?:widening|prominence)",
        "age-related atrophy", "cerebral and cerebellar atrophy", "diffuse atrophy",
    ],
    "Cerebellar degeneration": [
        "cerebellar degeneration", "cerebellar atrophy",
        "cerebellar volume loss", "cerebellar cortical atrophy",
    ],

    # 血管畸形 (1)
    "Cavernous hemangioma": [
        r"cavernous (?:h(?:a|e)mangioma|malformation)", "cavernoma",
    ],

    # 脱髓鞘 (1)
    "Demyelinating disease of central nervous system": [
        "demyelinat", "multiple sclerosis", "demyelinating disease",
        "demyelinating process",
    ],

    # 鞍区 (1)
    "Empty sella syndrome": [
        "empty sella", "partially empty sella", "empty sella turcica",
    ],

    # 脑室 (1)
    "Ventriculomegaly": [
        "ventriculomegaly",
        r"(?:mild |moderate |)(?:ventricular dilatation|ventricular enlargement)",
        r"ventricles?\s+(?:and|&)\s+.*?(?:are|appears?|is)\s+(?:widened|dilated|enlarged|prominent)",
        r"ventricles?\s*,?\s*(?:are|appears?|is)\s+(?:slightly\s+|mildly\s+|minimally\s+|markedly\s+|)(?:widened|dilated|enlarged|prominent|ectatic)",
        r"(?:lateral|third|4th|fourth|bilateral\s+lateral)\s+ventricles?\s*,?\s*(?:are|appears?|is)\s+(?:slightly\s+|mildly\s+|minimally\s+|markedly\s+|)(?:widened|dilated|enlarged|prominent|ectatic)",
        r"ventricles?\s+\w+\s+(?:appears?|are|is)\s+(?:widened|dilated|enlarged|prominent|ectatic)",
        r"enlargement\s+(?:is\s+present\s+in|of)\s+(?:both\s+|the\s+|)(?:lateral\s+|)(?:ventricles?)",
        r"ventricular\s+system\s+(?:is\s+|)(?:dilated|prominent|enlarged)",
        r"supratentorial\s+ventricles?\s*(?:and|&)\s*(?:subarachnoid|extra.axial)\s+spaces?\s*(?:are|appears?|is)\s*(?:widened|dilated|enlarged|prominent)",
        r"(?:mild |moderate |severe |)(?:tetraventricular |communicating |non.communicating |obstructive |)(?:hydrocephalus)",
        r"(?:lateral ventricles?|third ventricle|4th ventricle).{0,40}(?:dilated|enlarged|prominent|widened|ectatic)",
    ],

    # 感染 (2)
    "Mastoiditis": ["mastoiditis", "mastoid effusion", "mastoid (?:air cell |)disease"],
    "Chronic mastoiditis": ["chronic mastoiditis"],

    # 骨骼 (1)
    "Hyperostosis of skull": [
        "hyperostosis", "calvarial hyperostosis", "skull hyperostosis",
    ],

    # Chiari (1)
    "Chiari malformation": [
        "chiari", "tonsillar ectopia", "tonsillar herniation",
        "chiari malformation",
    ],

    # 脉络丛 (1)
    "Choroid plexus cyst": [
        "choroid plexus cyst", "choroid cyst", "choroid plexus xanthogranuloma",
    ],
}

LABEL_37_NAMES = list(LABEL_37_PATTERNS.keys())


# 否定句式
NEGATION_PATTERNS = [
    r"(?:was|is|are|were)\s+not\s+(?:detected|observed|seen|identified|found|present|evident|demonstrated|visualized|appreciated)",
    r"no\s+(?:evidence|sign|definite|convincing|clear|obvious)\s+(?:of|for)",
    r"without\s+(?:evidence|sign|definite)\s+(?:of|for)",
    r"no\s+(?:area|focus|region|lesion)\s+of\s+(?:restrict(?:ed|ion)\s+(?:compatible\s+with|consistent\s+with)?)",
    r"\bno\s+(?:\w+\s+){0,2}(?:acute\s+infarct|infarction|infarct)\b",
    r"negative\s+(?:for|finding)",
    r"ruled?\s+out",
    r"unlikely\s+to\s+(?:represent|be)",
    r"no\s+(?:significant|appreciable|identifiable|definite|overt)",
    r"no\s+suspicious",
]


def _check_negation(text, match_start, match_end):
    preceding_words = text[:match_start].split()[-3:]
    direct_neg = {"no", "not", "without", "negative"}
    if any(w in direct_neg for w in preceding_words):
        return True

    window_start = max(0, match_start - 80)
    window_end = min(len(text), match_end + 80)
    before = text[window_start:match_start]
    after = text[match_end:window_end]

    for neg_pat in NEGATION_PATTERNS:
        if re.search(neg_pat, before, re.IGNORECASE):
            return True
        if re.search(neg_pat, after, re.IGNORECASE):
            return True

    return False


# ============================================================================
# Label Extraction Functions
# ============================================================================
def extract_labels_37(report_text):
    """37 类关键词提取 + 否定检测"""
    text_lower = report_text.lower()
    labels = {}
    for label_name, patterns in LABEL_37_PATTERNS.items():
        found = False
        for pat in patterns:
            for match in re.finditer(pat, text_lower, re.IGNORECASE):
                if not _check_negation(text_lower, match.start(), match.end()):
                    found = True
                    break
            if found:
                break
        labels[label_name] = 1 if found else 0
    return labels


def extract_labels_37_from_file(input_path, output_path=None):
    """从 JSON 文件批量提取 37 类标签"""
    with open(input_path) as f:
        preds = json.load(f)

    results = []
    for p in preds:
        labels = extract_labels_37(p["report"])
        results.append(labels)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    total_pos = sum(sum(lb.values()) for lb in results)
    covered = sum(1 for n in LABEL_37_NAMES if any(lb[n] for lb in results))
    print(f"  37-label extraction: {len(results)} reports → {covered}/37 classes covered, {total_pos} positive labels")
    return results


# ============================================================================
# NLG Metrics
# ============================================================================
def compute_nlg(pred_reports, ref_reports):
    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_list, meteor_list, rouge_list = [], [], []
    for pred, ref in zip(pred_reports, ref_reports):
        pred_tok = pred.split()
        ref_tok = ref.split()
        bleu_list.append(sentence_bleu(
            [ref_tok], pred_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth))
        meteor_list.append(meteor_score([ref_tok], pred_tok))
        rouge_list.append(scorer.score(ref, pred)["rougeL"].fmeasure)

    return {
        "BLEU-4": float(np.mean(bleu_list)),
        "METEOR": float(np.mean(meteor_list)),
        "ROUGE-L": float(np.mean(rouge_list)),
    }


# ============================================================================
# Clinical F1
# ============================================================================
def _compute_f1_from_dicts(pred_labels, ref_labels, label_names):
    """通用：从 list[dict] 计算 Multi-label F1"""
    y_true = np.array([[lb[name] for name in label_names] for lb in ref_labels])
    y_pred = np.array([[lb[name] for name in label_names] for lb in pred_labels])

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    if pred_labels and pred_labels[0]:
        per_class = {}
        for name in label_names:
            if name not in ref_labels[0]:
                continue
            yt = np.array([lb[name] for lb in ref_labels])
            yp = np.array([lb[name] for lb in pred_labels])
            p = float(f1_score(yt, yp, average="binary", zero_division=0))
            r = float(f1_score(yt, yp, average="binary", zero_division=0))
            tp = int(((yp == 1) & (yt == 1)).sum())
            fp = int(((yp == 1) & (yt == 0)).sum())
            fn = int(((yp == 0) & (yt == 1)).sum())
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class[name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f, 4),
                "accuracy": round((tp + int(((yp == 0) & (yt == 0)).sum())) / max(len(yt), 1), 4),
                "support": support,
            }
    else:
        per_class = {}

    return {
        "Clinical_Macro_F1": macro,
        "Clinical_Micro_F1": micro,
        "per_pathology": per_class,
    }


def compute_clinical_f1_37(pred_labels, ref_labels):
    return _compute_f1_from_dicts(pred_labels, ref_labels, LABEL_37_NAMES)


# ============================================================================
# Diversity
# ============================================================================
def compute_diversity(pred_reports):
    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    normed = [normalize(r) for r in pred_reports]
    unique = len(set(normed))
    total = len(normed)
    mean_len = float(np.mean([len(r.split()) for r in pred_reports]))

    if total < 2:
        return {"uniqueness": 1.0, "duplicate_ratio": 0.0, "mean_len": mean_len, "most_common": []}

    freq = Counter(normed).most_common(3)

    return {
        "uniqueness": round(unique / total, 4),
        "duplicate_ratio": round(1.0 - unique / total, 4),
        "mean_len": round(mean_len, 1),
        "most_common": [(t, c) for t, c in freq],
    }


# ============================================================================
# Composite Score
# ============================================================================
def compute_composite(nlg, clinical, diversity):
    nlg_avg = (nlg["BLEU-4"] + nlg["METEOR"] + nlg["ROUGE-L"]) / 3
    clinical_mf1 = clinical["Clinical_Macro_F1"]
    diversity_score = diversity["uniqueness"]

    composite = WEIGHT_NLG * nlg_avg + WEIGHT_CLINICAL * clinical_mf1 + WEIGHT_DIVERSITY * diversity_score
    return round(composite, 4)


# ============================================================================
# Main Evaluation
# ============================================================================
def evaluate(preds, refs, gt_labels_path=None, no_clinical=False):
    pred_reports = [p["report"] for p in preds]
    ref_reports = [r["report"] for r in refs]

    print(f"\n{'=' * 70}")
    print(f"  MR-RATE Unified Evaluation v2")
    print(f"  Samples: {len(pred_reports)}")
    print(f"{'=' * 70}")

    # --- NLG ---
    print(f"\n  [NLG Metrics]  (weight = {WEIGHT_NLG})")
    nlg = compute_nlg(pred_reports, ref_reports)
    print(f"    BLEU-4:   {nlg['BLEU-4']:.4f}")
    print(f"    METEOR:   {nlg['METEOR']:.4f}")
    print(f"    ROUGE-L:  {nlg['ROUGE-L']:.4f}")
    nlg_avg = (nlg["BLEU-4"] + nlg["METEOR"] + nlg["ROUGE-L"]) / 3
    print(f"    NLG Avg:  {nlg_avg:.4f}")

    # --- Clinical (37 classes) ---
    clinical = None
    if not no_clinical:
        print(f"\n  [Clinical F1 — 37 Classes]  (weight = {WEIGHT_CLINICAL})")
        if gt_labels_path:
            with open(gt_labels_path) as f:
                gt_labels = json.load(f)
            pred_labels = [extract_labels_37(r["report"]) for r in preds]
        else:
            gt_labels = [extract_labels_37(r["report"]) for r in refs]
            pred_labels = [extract_labels_37(r["report"]) for r in preds]

        clinical = compute_clinical_f1_37(pred_labels, gt_labels)
        print(f"    Macro F1: {clinical['Clinical_Macro_F1']:.4f}")
        print(f"    Micro F1: {clinical['Clinical_Micro_F1']:.4f}")

        print(f"\n    Per-Pathology F1 (37 classes):")
        print(f"    {'Pathology':45s} {'F1':>7s} {'P':>7s} {'R':>7s} {'Supp':>5s}")
        print(f"    {'-'*70}")
        for name, pc in clinical.get("per_pathology", {}).items():
            if pc["support"] > 0:
                p_name = name[:44] if len(name) > 44 else name
                print(f"    {p_name:45s} {pc['f1']:7.4f} {pc['precision']:7.4f} {pc['recall']:7.4f} {pc['support']:5d}")

        zero_f1 = [n for n, pc in clinical.get("per_pathology", {}).items()
                   if pc["support"] > 0 and pc["f1"] == 0.0]
        if zero_f1:
            print(f"\n    F1=0 (need attention): {', '.join(zero_f1)}")

    # --- Diversity ---
    print(f"\n  [Diversity]  (weight = {WEIGHT_DIVERSITY})")
    diversity = compute_diversity(pred_reports)
    print(f"    Uniqueness:      {diversity['uniqueness']:.4f}")
    print(f"    Duplicate Ratio: {diversity['duplicate_ratio']:.4f}")
    print(f"    Mean Length:     {diversity['mean_len']:.1f} words")
    if diversity["most_common"]:
        print(f"    Most Common:     {diversity['most_common'][0][1]}x dup")

    # --- Composite ---
    if clinical is not None:
        composite = compute_composite(nlg, clinical, diversity)
        print(f"\n  {'='*70}")
        print(f"  [Composite Score]")
        print(f"  {WEIGHT_NLG} * {nlg_avg:.4f} + {WEIGHT_CLINICAL} * {clinical['Clinical_Macro_F1']:.4f} + {WEIGHT_DIVERSITY} * {diversity['uniqueness']:.4f}")
        print(f"  = {composite:.4f}")
        print(f"  {'='*70}")
    else:
        composite = None

    return {
        "num_samples": len(pred_reports),
        "nlg": nlg,
        "clinical": clinical,
        "diversity": diversity,
        "composite_score": composite,
    }


# ============================================================================
# Version Comparison
# ============================================================================
def compare_versions(results_list):
    print(f"\n{'=' * 70}")
    print(f"  Version Comparison")
    print(f"{'=' * 70}")
    header = f"{'Version':30s} | {'BLEU-4':>7s} | {'METEOR':>7s} | {'ROUGE-L':>7s} | {'C-Macro':>9s} | {'Uniq':>6s} | {'Score':>8s}"
    print(header)
    print("-" * len(header))

    best = {"composite_score": -1, "name": ""}
    for r in results_list:
        name = r.get("name", "unknown")[-30:]
        nlg = r["nlg"]
        clinical = r.get("clinical") or {}
        div = r["diversity"]
        cs = r.get("composite_score") or 0

        print(f"{name:30s} | {nlg['BLEU-4']:7.4f} | {nlg['METEOR']:7.4f} "
              f"| {nlg['ROUGE-L']:7.4f} | {clinical.get('Clinical_Macro_F1', -1):9.4f} "
              f"| {div['uniqueness']:5.1%} | {cs:8.4f}")

        if cs > best["composite_score"]:
            best = {"composite_score": cs, "name": name}

    print("-" * len(header))
    print(f"\n  Best: {best['name']} (Composite = {best['composite_score']:.4f})")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MR-RATE Unified Evaluation v2")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--refs", type=str, required=True)
    parser.add_argument("--gt-labels", type=str, default=None,
                        help="Ground truth 37-class labels JSON (if available)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--no-clinical", action="store_true",
                        help="Skip clinical metrics")
    parser.add_argument("--compare", type=str, nargs="*", default=None,
                        help="Additional prediction files for comparison")
    args = parser.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)
    with open(args.refs) as f:
        refs = json.load(f)

    result = evaluate(preds, refs, args.gt_labels, args.no_clinical)
    result["name"] = args.preds
    result["preds_file"] = args.preds

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n  Results saved → {out_path}")

    if args.compare:
        results = [result]
        for p in args.compare:
            with open(p) as f:
                preds_c = json.load(f)
            r = evaluate(preds_c, refs, args.gt_labels, args.no_clinical)
            r["name"] = p
            r["preds_file"] = p
            results.append(r)
        compare_versions(results)


if __name__ == "__main__":
    main()
