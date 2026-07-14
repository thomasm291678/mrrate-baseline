"""
MR-RATE LLM Evaluation via DeepSeek V4 Pro API
Replaces keyword-based label extraction with LLM-based extraction.
Tracks cost, latency, and accuracy.
"""
import os, sys, json, time, requests
import numpy as np
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")

API_KEY = "sk-9808455eb48749a5b7647f723a8a5632"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"

DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"
BATCH_FILTER = "batch27"

LABEL_NAMES = [
    "Cerebral infarction", "Cerebral hemorrhage", "Lacunar infarct",
    "Silent micro-hemorrhage of brain", "Subdural intracranial hemorrhage",
    "Intracranial aneurysm", "Watershed infarct",
    "Metastatic malignant neoplasm to brain", "Intracranial meningioma",
    "Schwannoma", "Glioma", "Pituitary adenoma", "Lipoma of brain",
    "Herniation of nucleus pulposus", "Spinal cord compression",
    "Hemangioma of vertebral column", "Spinal stenosis",
    "Foraminal Spinal Stenosis",
    "Arachnoid cyst", "Cyst of pineal gland", "Rathke's pouch cyst",
    "Mega cisterna magna", "Structure of cave of septum pellucidum",
    "Gliosis", "Encephalomalacia", "Cerebral edema",
    "Cerebral atrophy", "Cerebellar degeneration",
    "Cavernous hemangioma", "Demyelinating disease of central nervous system",
    "Empty sella syndrome", "Ventriculomegaly",
    "Mastoiditis", "Chronic mastoiditis",
    "Hyperostosis of skull", "Chiari malformation", "Choroid plexus cyst",
]

SYSTEM_PROMPT = """You are an expert neuroradiologist evaluating brain/spine MRI reports. Your task is to extract exactly which of 37 pathologies are POSITIVELY documented in a given report.

## Decision Framework

### POSITIVE (label = 1) — a finding is confirmed or reasonably suspected:
- Explicitly stated as present: "there is a lacunar infarct", "meningioma is identified"
- Descriptive language implying presence: "compatible with", "consistent with", "in keeping with", "findings suggest"
- Tentative/uncertain but actively considered: "possible", "suspicious for", "cannot exclude", "concerning for"
- Incidental findings that are clearly observed: "incidental note is made of an arachnoid cyst"
- Generally described as present even without the exact disease name (e.g. "acute ischemia with diffusion restriction" → Cerebral infarction = 1)
- Stable findings that are still present: "stable left frontal encephalomalacia" → Encephalomalacia = 1
- Findings with qualifiers like "mild", "minimal", "subtle" that are explicitly observed

### NEGATIVE (label = 0) — absent, ruled out, or purely historical:
- Explicit negation: "no evidence of", "not seen", "not identified", "negative for"
- Ruled-out diagnoses: "ruled out", "unlikely", "not suspected"
- Historical only: "history of glioblastoma" without confirmation of current disease
- Post-operative changes without residual disease: "post-operative cavity, no residual tumor enhancement"
- Absence of finding despite suspicion: "no restricted diffusion to suggest acute infarct"
- Age-related involutional changes: "age-appropriate cerebral atrophy" → Cerebral atrophy = 0
- Normal variants or anatomical descriptions without pathology: "prominent cisterna magna" without "mega cisterna magna" diagnosis
- Findings described as "within normal limits" or "unremarkable"

## Output Format

Return ONLY a JSON object. No markdown fences, no explanations, no preamble.

### Example:
{"Cerebral infarction": 0, "Cerebral hemorrhage": 0, "Lacunar infarct": 1, "Silent micro-hemorrhage of brain": 0, "Subdural intracranial hemorrhage": 0, "Intracranial aneurysm": 0, "Watershed infarct": 0, "Metastatic malignant neoplasm to brain": 0, "Intracranial meningioma": 1, "Schwannoma": 0, "Glioma": 0, "Pituitary adenoma": 0, "Lipoma of brain": 0, "Herniation of nucleus pulposus": 0, "Spinal cord compression": 0, "Hemangioma of vertebral column": 0, "Spinal stenosis": 0, "Foraminal Spinal Stenosis": 0, "Arachnoid cyst": 0, "Cyst of pineal gland": 0, "Rathke's pouch cyst": 0, "Mega cisterna magna": 0, "Structure of cave of septum pellucidum": 0, "Gliosis": 1, "Encephalomalacia": 0, "Cerebral edema": 0, "Cerebral atrophy": 0, "Cerebellar degeneration": 0, "Cavernous hemangioma": 0, "Demyelinating disease of central nervous system": 0, "Empty sella syndrome": 0, "Ventriculomegaly": 0, "Mastoiditis": 0, "Chronic mastoiditis": 0, "Hyperostosis of skull": 0, "Chiari malformation": 0, "Choroid plexus cyst": 0}

## Pathology Categories (for context — report may discuss brain, spine, or both)

### Cerebrovascular (7)
Cerebral infarction, Cerebral hemorrhage, Lacunar infarct, Silent micro-hemorrhage of brain, Subdural intracranial hemorrhage, Intracranial aneurysm, Watershed infarct

### Neoplasms (6)
Metastatic malignant neoplasm to brain, Intracranial meningioma, Schwannoma, Glioma, Pituitary adenoma, Lipoma of brain

### Spine (5)
Herniation of nucleus pulposus, Spinal cord compression, Hemangioma of vertebral column, Spinal stenosis, Foraminal Spinal Stenosis

### Congenital / Benign Variants (5)
Arachnoid cyst, Cyst of pineal gland, Rathke's pouch cyst, Mega cisterna magna, Structure of cave of septum pellucidum

### Brain Parenchyma (3)
Gliosis, Encephalomalacia, Cerebral edema

### Degenerative (2)
Cerebral atrophy, Cerebellar degeneration

### Vascular Malformation (1)
Cavernous hemangioma

### Demyelinating (1)
Demyelinating disease of central nervous system

### Sellar Region (1)
Empty sella syndrome

### Ventricular (1)
Ventriculomegaly

### Infectious / Inflammatory (2)
Mastoiditis, Chronic mastoiditis

### Skeletal / Congenital (2)
Hyperostosis of skull, Chiari malformation

### Choroid Plexus (1)
Choroid plexus cyst

## Important Notes
- Most reports will have the majority of labels as 0 — this is normal and expected
- If a brain-only report discusses no spine findings, all 5 spine labels should be 0
- Do NOT infer findings not explicitly mentioned or reasonably implied by the text
- When uncertain between 0 and 1 for a borderline finding, lean toward 0"""

COST_TRACKER = {"total_tokens": 0, "total_cost": 0.0, "total_time": 0.0, "calls": 0}


def load_reports(split, n=None):
    splits_df = pd.read_csv(f"{DATA}/splits.csv")
    samples = splits_df[(splits_df["split"] == split) & (splits_df["batch_id"] == BATCH_FILTER)]
    if n:
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


def call_deepseek(report_text, retries=3):
    user_msg = f"Report:\n{report_text[:3000]}"

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
                timeout=90,
            )
            elapsed = time.time() - t0
            data = resp.json()

            if "error" in data:
                print(f"  API error: {data['error']}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, elapsed, 0, 0

            usage = data.get("usage", {})
            in_tokens = usage.get("prompt_tokens", 0)
            out_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # DeepSeek V4 Pro pricing
            cost = (in_tokens / 1_000_000) * 2.0 + (out_tokens / 1_000_000) * 8.0

            COST_TRACKER["total_tokens"] += total_tokens
            COST_TRACKER["total_cost"] += cost
            COST_TRACKER["total_time"] += elapsed
            COST_TRACKER["calls"] += 1

            msg = (data.get("choices", [{}])[0].get("message", {}) or {})
            content = (msg.get("content", "") or "")
            reasoning = (msg.get("reasoning_content", "") or "")

            # If content is empty, try to extract JSON from the end of reasoning
            if not content and reasoning:
                # Find last complete JSON object in reasoning
                last_brace = reasoning.rfind("}")
                if last_brace > 0:
                    first_brace = reasoning.rfind("{", 0, last_brace)
                    if first_brace >= 0:
                        content = reasoning[first_brace:last_brace+1]

            if not content:
                print(f"  No content/reasoning extractable")
                return None, elapsed, in_tokens, out_tokens
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

            try:
                labels = json.loads(content)
            except json.JSONDecodeError:
                import re
                m = re.search(r'\{[^{}]*"Cerebral infarction"[^{}]*\}', content, re.DOTALL)
                if m:
                    labels = json.loads(m.group())
                else:
                    m2 = re.search(r'\{.*\}', content, re.DOTALL)
                    if m2:
                        try:
                            labels = json.loads(m2.group())
                        except:
                            print(f"  Failed to parse: {content[:200]}")
                            if attempt < retries - 1:
                                time.sleep(2 ** attempt)
                                continue
                            return None, elapsed, in_tokens, out_tokens
                    else:
                        print(f"  No JSON found: {content[:200]}")
                        if attempt < retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None, elapsed, in_tokens, out_tokens

            # Validate all 37 labels exist
            for name in LABEL_NAMES:
                if name not in labels:
                    labels[name] = 0

            return labels, elapsed, in_tokens, out_tokens

        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt+1})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, 0, 0, 0
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, 0, 0, 0

    return None, 0, 0, 0


def process_reports(reports, split_name):
    labels_list = []
    total_start = time.time()

    print(f"\n--- Processing {len(reports)} reports ({split_name}) ---")
    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(call_deepseek, r): i for i, r in enumerate(reports)}
        for future in as_completed(futures):
            i = futures[future]
            labels_result = future.result()
            if labels_result[0] is not None:
                labels, elapsed, in_tok, out_tok = labels_result
            else:
                labels, elapsed, in_tok, out_tok = None, 0, 0, 0
            labels_list.append((i, labels, elapsed, in_tok, out_tok))

            done = len(labels_list)
            if done % 10 == 0 or done == len(reports):
                elapsed_total = time.time() - total_start
                speed = done / elapsed_total if elapsed_total > 0 else 0
                print(f"  [{done}/{len(reports)}] {speed:.1f} req/s | "
                      f"tokens: {COST_TRACKER['total_tokens']:,} | "
                      f"cost: ¥{COST_TRACKER['total_cost']:.4f}")

    labels_list.sort(key=lambda x: x[0])
    labels_llm = [lb[1] for lb in labels_list]
    total_elapsed = time.time() - total_start
    failed = sum(1 for lb in labels_llm if lb is None)
    success = len(labels_llm) - failed

    print(f"\n  [{split_name}] {success} ok, {failed} failed | {total_elapsed:.1f}s")
    return labels_llm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, help="Run N test samples only")
    parser.add_argument("--max-calls", type=int, default=0, help="Total API calls budget (split across val preds+refs, test preds+refs)")
    parser.add_argument("--split", type=str, default="val", help="Single split mode")
    args = parser.parse_args()

    if args.test:
        print(f"[TEST MODE] {args.test} samples")
        reports = load_reports(args.split, n=args.test)
        process_reports(reports, args.split)
        return

    if args.max_calls > 0:
        max_calls = args.max_calls
        # val: 190 preds + 190 refs = 380 calls
        # remaining for test = max_calls - 380, split equally preds/refs
        val_n = 190
        val_calls = val_n * 2
        remaining = max_calls - val_calls
        if remaining < 0:
            val_n = max_calls // 2
            val_calls = val_n * 2
            test_n = 0
        else:
            test_n = min(300, remaining // 2)
        test_calls = test_n * 2

        print(f"=== Dual-mode LLM Evaluation (max {max_calls} calls) ===")
        print(f"  val:  {val_n} preds + {val_n} refs = {val_calls} calls")
        print(f"  test: {test_n} preds + {test_n} refs = {test_calls} calls")
        print(f"  total calls: {val_calls + test_calls}")
        est_cost = (val_calls + test_calls) * 0.012
        print(f"  est. cost: ¥{est_cost:.2f}")
        print(f"  model: {MODEL}")
        print()

        out_dir = Path("/home/jiaqigu/mrrate_hidnet/outputs/report_gen")

        # --- val preds ---
        print("[1/4] val preds")
        val_preds = load_reports("val", n=val_n)
        val_pred_labels = process_reports(val_preds, "val_preds")
        json.dump(val_pred_labels, open(out_dir / "llm_labels_val_preds.json", "w"), indent=2)

        # --- val refs (same reports) ---
        COST_TRACKER["total_tokens"] = 0
        COST_TRACKER["total_cost"] = 0.0
        print("\n[2/4] val refs")
        val_ref_labels = process_reports(val_preds, "val_refs")
        json.dump(val_ref_labels, open(out_dir / "llm_labels_val_refs.json", "w"), indent=2)

        # --- test preds ---
        if test_n > 0:
            COST_TRACKER["total_tokens"] = 0
            COST_TRACKER["total_cost"] = 0.0
            print("\n[3/4] test preds")
            test_preds = load_reports("test", n=test_n)
            test_pred_labels = process_reports(test_preds, "test_preds")
            json.dump(test_pred_labels, open(out_dir / "llm_labels_test_preds.json", "w"), indent=2)

            COST_TRACKER["total_tokens"] = 0
            COST_TRACKER["total_cost"] = 0.0
            print("\n[4/4] test refs")
            test_ref_labels = process_reports(test_preds, "test_refs")
            json.dump(test_ref_labels, open(out_dir / "llm_labels_test_refs.json", "w"), indent=2)

        print(f"\n{'='*60}")
        print(f"  LLM Dual Evaluation Done")
        print(f"  Files saved to {out_dir}/")
        print(f"  llm_labels_val_preds.json, llm_labels_val_refs.json")
        if test_n > 0:
            print(f"  llm_labels_test_preds.json, llm_labels_test_refs.json")
        print(f"{'='*60}")
        return

    # Single split mode
    reports = load_reports(args.split)
    print(f"  Samples: {len(reports)}")
    est_cost = len(reports) * 0.012
    print(f"  Est. cost: ¥{est_cost:.2f}")
    process_reports(reports, args.split)


if __name__ == "__main__":
    main()
