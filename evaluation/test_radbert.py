"""
Test RadBERT on 10 MRI reports — reads reports directly from CSV (no image loading).
If MRI F1 < 0.4, also test on CT-RATE for comparison.
"""
import os, sys, torch, numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")

RADBERT = "microsoft/BiomedVLP-CXR-BERT-specialized"
DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"
N_TRAIN = 50
N_TEST = 10
BATCH = 8

def load_eval_labels():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_v2", "/home/jiaqigu/mrrate_hidnet/evaluation_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LABEL_37_NAMES, mod.extract_labels_37

def sample_reports(n, split="test", batch_filter="batch27"):
    splits_df = pd.read_csv(f"{DATA}/splits.csv")
    samples = splits_df[(splits_df["split"] == split) & (splits_df["batch_id"] == batch_filter)]
    samples = samples.head(n)
    
    reports = {}
    batch_ids = samples["batch_id"].unique()
    for bid in batch_ids:
        rp = f"{DATA}/reports/{bid}_reports.csv"
        if Path(rp).exists():
            rdf = pd.read_csv(rp)
            for _, row in rdf.iterrows():
                reports[str(row["study_uid"])] = str(row.get("report", ""))
    
    result = []
    for _, row in samples.iterrows():
        r = reports.get(str(row["study_uid"]), "")
        if r.strip():
            result.append(r.strip())
    return result

def embed_with_radbert(texts):
    from transformers import AutoTokenizer, AutoModel
    print("  Loading RadBERT tokenizer+model from HuggingFace...")
    tok = AutoTokenizer.from_pretrained(RADBERT, trust_remote_code=True)
    model = AutoModel.from_pretrained(RADBERT, trust_remote_code=True).to("cuda")
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            inputs = tok(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to("cuda")
            cls = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls)
            torch.cuda.empty_cache()
    return np.concatenate(embs, axis=0)

def evaluate_radbert(reports, label_names, extract_fn, tag=""):
    print(f"  Extracting labels for {len(reports)} reports...")
    gt_labels = np.array([[extract_fn(r)[name] for name in label_names] for r in reports])
    
    print(f"  Embedding with RadBERT...")
    emb = embed_with_radbert(reports)
    
    print(f"  Training per-label probes...")
    f1s = {}
    for i, name in enumerate(label_names):
        y = gt_labels[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            f1s[name] = 0.0
            continue
        try:
            Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.3,
                                                    stratify=y if min(y.sum(), len(y)-y.sum()) >= 2 else None,
                                                    random_state=42)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.3, random_state=42)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            f1s[name] = 0.0
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)
        f1s[name] = f1_score(yte, yp, zero_division=0)

    macro = np.mean(list(f1s.values()))
    non_zero = sum(1 for v in f1s.values() if v > 0)
    print(f"\n--- RadBERT Probe ({tag}) ---")
    print(f"  Samples: {len(reports)}")
    print(f"  Macro F1: {macro:.4f}")
    print(f"  Non-zero labels: {non_zero}/{len(label_names)}")

    sorted_f1 = sorted(f1s.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5: {sorted_f1[:5]}")
    print(f"  Bottom 5: {sorted_f1[-5:]}")
    return macro, f1s

def main():
    import sys
    label_names, extract_fn = load_eval_labels()
    print(f"Labels: {len(label_names)}")
    print(f"RadBERT: {RADBERT}")

    print(f"\n{'='*50}")
    print("=== MRI test ===")
    print(f"Loading {N_TRAIN} train + {N_TEST} test reports...")
    mri_train = sample_reports(N_TRAIN, "train")
    mri_test = sample_reports(N_TEST, "test")
    all_mri = mri_train + mri_test
    print(f"  Got {len(mri_train)} train + {len(mri_test)} test = {len(all_mri)} reports")
    mri_macro, mri_f1s = evaluate_radbert(all_mri, label_names, extract_fn, "MRI reports")

    if mri_macro < 0.4:
        print(f"\nMRI F1 ({mri_macro:.4f}) < 0.4, checking CT dataset...")
        ct_paths = [
            Path("/mnt/nas1/disk07/public/CT-RATE"),
            Path("/mnt/nas1/disk07/public/ct_rate"),
            Path("/mnt/nas1/disk07/public/CT_data/CT-RATE"),
        ]
        ct_found = None
        for p in ct_paths:
            if p.exists():
                ct_found = p
                break

        if ct_found:
            print(f"CT dataset found: {ct_found}")
            reports_csv = ct_found / "reports"
            csv_files = list(reports_csv.glob("*.csv")) if reports_csv.exists() else []
            if csv_files:
                df = pd.concat([pd.read_csv(f, nrows=5) for f in csv_files[:3]])
                ct_texts = df["report"].dropna().tolist()[:N_TRAIN + N_TEST]
                if len(ct_texts) >= 10:
                    ct_macro, ct_f1s = evaluate_radbert(ct_texts, label_names, extract_fn, "CT reports")
                    print(f"\n{'='*50}")
                    print(f"  MRI Macro F1: {mri_macro:.4f}")
                    print(f"  CT  Macro F1: {ct_macro:.4f}")
                    if ct_macro > mri_macro * 1.5:
                        print(f"  => RadBERT is significantly better on CT. MRI needs domain adaptation.")
                    else:
                        print(f"  => Similar performance. RadBERT transfers reasonably to MRI.")
                    sys.exit(0)
            print(f"  No CT CSV reports found in {ct_found}")
        else:
            print("  CT-RATE dataset not found on NAS")

    print("\nDone.")

if __name__ == "__main__":
    main()
