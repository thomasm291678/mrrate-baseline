import torch
import numpy as np
from collections import Counter
from evaluate import load as load_metric
from sklearn.feature_extraction.text import TfidfVectorizer


_bleu4 = load_metric("bleu")
_rouge = load_metric("rouge")
_meteor = load_metric("meteor")
_bleu1 = load_metric("bleu")

try:
    _bertscore = load_metric("bertscore")
    _has_bertscore = True
except Exception:
    _has_bertscore = False


def compute_metrics(preds, refs):
    refs_wrapped = [[r] for r in refs]

    result = {}
    result["bleu1"] = _bleu1.compute(
        predictions=preds, references=refs_wrapped, max_order=1)["bleu"]
    result["bleu4"] = _bleu4.compute(
        predictions=preds, references=refs_wrapped, max_order=4)["bleu"]
    result["rougeL"] = _rouge.compute(
        predictions=preds, references=refs)["rougeL"]
    result["meteor"] = _meteor.compute(
        predictions=preds, references=refs)["meteor"]

    if _has_bertscore:
        try:
            bs = _bertscore.compute(
                predictions=preds, references=refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")
            result["bert_f1"] = float(np.mean(bs["f1"]))
        except Exception:
            result["bert_f1"] = -1.0
    else:
        result["bert_f1"] = -1.0

    return result


def compute_f1(pred, gt):
    common = Counter(pred.split()) & Counter(gt.split())
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred.split())
    recall = num_same / len(gt.split())
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def template_collapse_stats(preds):
    if len(preds) < 2:
        return {"unique_count": len(preds), "total": len(preds),
                "uniqueness": 1.0, "duplicate_ratio": 0.0,
                "mean_len": len(preds[0]) if preds else 0,
                "pairwise_sim": 0.0}

    unique = len(set(preds))
    total = len(preds)
    mean_len = np.mean([len(p.split()) for p in preds])

    try:
        tfidf = TfidfVectorizer().fit_transform(preds)
        sim = (tfidf * tfidf.T).toarray()
        n = sim.shape[0]
        upper = sim[np.triu_indices(n, k=1)]
        pair_sim = float(np.mean(upper))
    except Exception:
        pair_sim = 0.0

    freq = Counter(preds)
    dup_ratio = 1.0 - unique / total if total > 0 else 0.0

    return {"unique_count": unique, "total": total,
            "uniqueness": unique / total, "duplicate_ratio": dup_ratio,
            "mean_len": float(mean_len), "pairwise_sim": pair_sim,
            "top3_freq": freq.most_common(3)}


@torch.no_grad()
def run_eval(enc, llm, tok, val_loader, dev, n_vt,
             max_samples=200, max_new_tokens=256):
    enc.eval()
    llm.eval()

    all_preds = []
    all_refs = []

    for _, batch in enumerate(val_loader):
        if len(all_preds) >= max_samples:
            break

        t1 = batch["t1"].to(dev)
        flair = batch["flair"].to(dev)
        t2 = batch["t2"].to(dev)
        h1, hf, h2 = batch["has_t1"], batch["has_flair"], batch["has_t2"]
        B = t1.shape[0]
        reports = batch["reports"]

        vt = enc(t1, flair, t2, h1, hf, h2)

        prefix_ids = tok.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
            return_tensors="pt").expand(B, -1).to(dev)

        prefix_embeds = llm.get_input_embeddings()(prefix_ids)
        combined = torch.cat([vt.to(prefix_embeds.dtype), prefix_embeds], dim=1)

        generated = llm.generate(
            inputs_embeds=combined,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        for i in range(B):
            gen_ids = generated[i]
            pt_len = n_vt + prefix_ids.shape[1]
            out_ids = gen_ids[pt_len:]
            pred = tok.decode(out_ids, skip_special_tokens=True)
            all_preds.append(pred.strip())
            all_refs.append(reports[i].strip())

    enc.train()
    llm.train()

    if not all_preds:
        return {"error": "no predictions"}, [], []

    metrics = compute_metrics(all_preds, all_refs)
    metrics.update(template_collapse_stats(all_preds))

    return metrics, all_preds, all_refs
