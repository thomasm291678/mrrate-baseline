"""
Compare LLM (DeepSeek V4 Pro) vs Keyword label extraction on MRI reports.
Outputs: per-class F1 comparison, macro/micro summary, agreement stats.
"""
import sys, json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "eval_v2", "/home/jiaqigu/mrrate_hidnet/evaluation_v2.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
LABEL_NAMES = mod.LABEL_37_NAMES
extract_keyword = mod.extract_labels_37

OUT = Path("/home/jiaqigu/mrrate_hidnet/outputs/report_gen")

FILES = {
    "val_preds": OUT / "llm_labels_val_preds.json",
    "val_refs": OUT / "llm_labels_val_refs.json",
    "test_preds": OUT / "llm_labels_test_preds.json",
    "test_refs": OUT / "llm_labels_test_refs.json",
}

import pandas as pd

DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"

def load_reports(split, n):
    splits_df = pd.read_csv(f"{DATA}/splits.csv")
    samples = splits_df[(splits_df["split"] == split) & (splits_df["batch_id"] == "batch27")]
    samples = samples.head(n)
    reports_map = {}
    for bid in samples["batch_id"].unique():
        rp = f"{DATA}/reports/{bid}_reports.csv"
        if Path(rp).exists():
            rdf = pd.read_csv(rp)
            for _, row in rdf.iterrows():
                reports_map[str(row["study_uid"])] = str(row.get("report", ""))
    result = []
    for _, row in samples.iterrows():
        r = reports_map.get(str(row["study_uid"]), "")
        if r.strip():
            result.append(r.strip())
    return result

def load_llm(path):
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    valid = [d for d in data if d is not None]
    if len(valid) != len(data):
        print(f"  Warning: {len(data)-len(valid)} None entries in {path.name}")
    return valid

def labels_to_matrix(labels_list):
    return np.array([[lb.get(name, 0) for name in LABEL_NAMES] for lb in labels_list])

def compute_pairwise(y_true, y_pred):
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if yt.sum() == 0 and yp.sum() == 0:
            per_class[name] = float("nan")
            continue
        per_class[name] = float(f1_score(yt, yp, average="binary", zero_division=0))
    return macro, micro, per_class

def compute_agreement(kw_matrix, llm_matrix):
    agreements = []
    for i in range(kw_matrix.shape[1]):
        yk = kw_matrix[:, i]
        yl = llm_matrix[:, i]
        if yk.sum() == 0 and yl.sum() == 0:
            agreements.append(1.0)
        elif len(np.unique(yk)) < 2 or len(np.unique(yl)) < 2:
            agreements.append(float((yk == yl).mean()))
        else:
            try:
                agreements.append(float(cohen_kappa_score(yk, yl)))
            except:
                agreements.append(float((yk == yl).mean()))
    return np.mean(agreements), agreements

def compare(split, n, preds_llm_path, refs_llm_path, tag):
    print(f"\n{'='*70}")
    print(f"  {tag} — LLM vs Keyword Comparison ({n} samples)")
    print(f"{'='*70}")

    reports = load_reports(split, n)
    print(f"  Reports loaded: {len(reports)}")

    preds_llm = load_llm(preds_llm_path)
    refs_llm = load_llm(refs_llm_path)
    if preds_llm is None or refs_llm is None:
        return
    n = min(len(reports), len(preds_llm), len(refs_llm))
    reports = reports[:n]
    preds_llm = preds_llm[:n]
    refs_llm = refs_llm[:n]

    # Keyword labels
    print(f"  Extracting keyword labels...")
    preds_kw = [extract_keyword(r) for r in reports]
    refs_kw = [extract_keyword(r) for r in reports]

    m_kw = labels_to_matrix(preds_kw)
    m_llm = labels_to_matrix(preds_llm)
    m_ref_kw = labels_to_matrix(refs_kw)
    m_ref_llm = labels_to_matrix(refs_llm)

    # Clinical F1: keyword vs LLM on reference
    ref_macro_kw_kw, ref_micro_kw_kw, _ = compute_pairwise(m_ref_kw, m_ref_kw)
    ref_macro_llm_kw, ref_micro_llm_kw, ref_per = compute_pairwise(m_ref_kw, m_ref_llm)

    # Clinical F1 on predictions
    pred_macro_kw_kw, pred_micro_kw_kw, _ = compute_pairwise(m_kw, m_kw)
    pred_macro_llm_kw, pred_micro_llm_kw, pred_per = compute_pairwise(m_kw, m_llm)

    # Agreement: keyword vs LLM
    pred_agree, pred_agree_per = compute_agreement(m_kw, m_llm)
    ref_agree, ref_agree_per = compute_agreement(m_ref_kw, m_ref_llm)

    print(f"\n  --- Reference Labels (ground truth comparison) ---")
    print(f"  LLM vs Keyword Macro F1: {ref_macro_llm_kw:.4f}")
    print(f"  LLM vs Keyword Micro F1: {ref_micro_llm_kw:.4f}")
    print(f"  Avg Cohen's Kappa:      {ref_agree:.4f}")

    print(f"\n  --- Predicted Labels (model output comparison) ---")
    print(f"  LLM vs Keyword Macro F1: {pred_macro_llm_kw:.4f}")
    print(f"  LLM vs Keyword Micro F1: {pred_micro_llm_kw:.4f}")
    print(f"  Avg Cohen's Kappa:      {pred_agree:.4f}")

    # Show big disagreements
    print(f"\n  --- Top Disagreements (reference) ---")
    ref_diffs = [(name, ref_per[name]) for name in LABEL_NAMES
                 if not np.isnan(ref_per.get(name, float('nan'))) and ref_per.get(name, 0) < 0.5]
    ref_diffs.sort(key=lambda x: x[1])
    for name, f1 in ref_diffs[:10]:
        print(f"    {name:50s} F1={f1:.4f}")

    # Summary counts
    kw_positive = m_ref_kw.sum()
    llm_positive = m_ref_llm.sum()
    print(f"\n  --- Label Density ---")
    print(f"  Keyword positive labels: {kw_positive} ({kw_positive/(n*37)*100:.1f}%)")
    print(f"  LLM     positive labels: {llm_positive} ({llm_positive/(n*37)*100:.1f}%)")
    print(f"  LLM/keyword ratio:       {llm_positive/kw_positive:.2f}x" if kw_positive > 0 else "  LLM/keyword ratio: N/A")

    return {
        "split": split, "n": n,
        "ref_macro_f1": ref_macro_llm_kw, "ref_kappa": ref_agree,
        "pred_macro_f1": pred_macro_llm_kw, "pred_kappa": pred_agree,
        "kw_positive": int(kw_positive), "llm_positive": int(llm_positive),
    }


def main():
    print("="*70)
    print("  MR-RATE: LLM vs Keyword Label Extraction Comparison")
    print("="*70)

    results = []

    r1 = compare("val", 186, FILES["val_preds"], FILES["val_refs"], "VAL")
    if r1: results.append(r1)

    r2 = compare("test", 57, FILES["test_preds"], FILES["test_refs"], "TEST")
    if r2: results.append(r2)

    print(f"\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  [{r['split'].upper()}] n={r['n']}")
        print(f"    LLM vs Keyword (ref)  Macro F1: {r['ref_macro_f1']:.4f}  Kappa: {r['ref_kappa']:.4f}")
        print(f"    LLM vs Keyword (pred) Macro F1: {r['pred_macro_f1']:.4f}  Kappa: {r['pred_kappa']:.4f}")
        ratio = r['llm_positive'] / r['kw_positive'] if r['kw_positive'] > 0 else 0
        print(f"    Positive labels: keyword={r['kw_positive']}  llm={r['llm_positive']}  ratio={ratio:.2f}x")

    json.dump(results, open(OUT / "llm_vs_keyword_summary.json", "w"), indent=2)
    print(f"\n  Summary saved -> {OUT / 'llm_vs_keyword_summary.json'}")

if __name__ == "__main__":
    main()
