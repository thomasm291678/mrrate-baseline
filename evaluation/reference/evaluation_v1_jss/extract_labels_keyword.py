#!/usr/bin/env python3
"""
极简 label 提取器 —— 关键词匹配 + 否定检测。
不依赖任何模型，5 分钟跑通，用于本地评估近似官方 Clinical F1。

用法:
  python extract_labels_keyword.py preds.json -o pred_labels.json
  python extract_labels_keyword.py preds.json                 # 默认输出 pred_labels.json
"""
import json
import re
import argparse

# 37 类标签 → 匹配关键词列表（支持同义词/变体）
LABEL_PATTERNS = {
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
        "ex vacuo dilatation",
        "sulcal (?:widening|prominence)", "age-related atrophy",
        "cerebral and cerebellar atrophy", "diffuse atrophy",
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


# 特定否定句式（匹配整个短语）
NEGATION_PATTERNS = [
    # 后置否定: "X was not detected/observed/seen/identified/found"
    r"(?:was|is|are|were)\s+not\s+(?:detected|observed|seen|identified|found|present|evident|demonstrated|visualized|appreciated)",
    # 前置否定: "no evidence/sign of X", "without evidence of X"
    r"no\s+(?:evidence|sign|definite|convincing|clear|obvious)\s+(?:of|for)",
    r"without\s+(?:evidence|sign|definite)\s+(?:of|for)",
    # 放射报告特殊句式: "no ... compatible with / consistent with ... was observed"
    # "no area of restricted diffusion" / "no area of restriction compatible with"
    r"no\s+(?:area|focus|region|lesion)\s+of\s+(?:restrict(?:ed|ion)\s+(?:compatible\s+with|consistent\s+with)?)",
    # 简短前置否定: "no acute infarct", "no infarction" (匹配词前最多 3 个词内有 no)
    r"\bno\s+(?:\w+\s+){0,2}(?:acute\s+infarct|infarction|infarct)\b",
    # 直接否定: "negative for X"
    r"negative\s+(?:for|finding)",
    # 排除: "ruled out"
    r"ruled?\s+out",
    # 不典型: "unlikely to represent"
    r"unlikely\s+to\s+(?:represent|be)",
    # 未见/无明显
    r"no\s+(?:significant|appreciable|identifiable|definite|overt)",
    r"no\s+suspicious",
]


def _check_negation(text, match_start, match_end):
    """
    检查匹配位置前后是否存在特定否定句式。
    三层检测:
      1. 匹配前 3 个词内是否有直接否定词 (no, not, without)
      2. 匹配前后 80 字符窗口内是否有特定否定句式
      3. 匹配后紧跟的否定结构 ("was not detected")
    """
    # Layer 1: 直接前置否定词 (匹配前最多 3 个词)
    preceding_words = text[:match_start].split()[-3:]
    direct_neg = {"no", "not", "without", "negative"}
    if any(w in direct_neg for w in preceding_words):
        return True

    # Layer 2: 80 字符窗口内的特定否定句式
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


def extract_labels_from_report(report_text):
    """
    输入: 一段报告文字
    输出: dict {label_name: 0/1}  37 类
    """
    text_lower = report_text.lower()

    labels = {}
    for label_name, patterns in LABEL_PATTERNS.items():
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


def main():
    ap = argparse.ArgumentParser(
        description="极简 label 提取器 —— 关键词匹配 + 否定检测")
    ap.add_argument("input", help="预测报告 JSON 文件路径")
    ap.add_argument("-o", "--output", default="pred_labels.json",
                    help="输出标签 JSON 路径 (默认: pred_labels.json)")
    args = ap.parse_args()

    with open(args.input) as f:
        preds = json.load(f)

    results = []
    for p in preds:
        labels = extract_labels_from_report(p["report"])
        results.append(labels)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # 统计
    total_pos = sum(sum(lb.values()) for lb in results)
    covered = sum(1 for n in LABEL_PATTERNS if any(lb[n] for lb in results))
    label_pos_counts = {n: sum(lb[n] for lb in results) for n in LABEL_PATTERNS}
    active_labels = [(n, c) for n, c in label_pos_counts.items() if c > 0]
    active_labels.sort(key=lambda x: -x[1])

    print(f"提取完成: {len(results)} 条报告 → {args.output}")
    print(f"总阳性标签数: {total_pos}，覆盖 {covered}/37 类")
    if active_labels:
        print("阳性标签分布 (top 10):")
        for name, count in active_labels[:10]:
            print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
