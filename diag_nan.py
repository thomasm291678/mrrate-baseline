import torch, torch.nn as nn, sys
sys.path.insert(0, ".")
from encoder_v5 import ReportingModelV5
from mrrate_dataset import MRRateDataset

dev = torch.device("cuda")
enc = ReportingModelV5(llm_dim=2048, grid=2, base_ch=32, n_vt_out=1).to(dev)

# Load Phase 1
ck = torch.load("outputs/report_gen/phase1_latest.pt", map_location=dev, weights_only=False)
enc.load_state_dict(ck["encoder_state"], strict=False)

for p in enc.t1_enc.parameters(): p.requires_grad = False
for p in enc.t2_enc.parameters(): p.requires_grad = False
for p in enc.flair_enc.parameters(): p.requires_grad = False

import json, os
from safetensors.torch import load_file
QWEN = "/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct"
idx = json.load(open(os.path.join(QWEN, "model.safetensors.index.json")))
p = idx["weight_map"]["model.embed_tokens.weight"]
emb_w = load_file(os.path.join(QWEN, p))["model.embed_tokens.weight"].to(dev).bfloat16()
emb = nn.Embedding.from_pretrained(emb_w)
emb.weight.requires_grad = False

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

ds = MRRateDataset("/mnt/nas1/disk07/public/mr_data/MR-RATE", "train", batch_filter="batch27")
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

freq_weights = torch.tensor([
    0.30, 0.05, 0.25, 0.03, 0.02, 0.01, 0.02,
    0.05, 0.08, 0.02, 0.10, 0.03, 0.005,
    0.20, 0.08, 0.02, 0.15, 0.08, 0.04,
    0.06, 0.05, 0.10, 0.02, 0.04,
    0.03, 0.02, 0.04, 0.01, 0.01, 0.03,
    0.40, 0.15, 0.05, 0.06, 0.02, 0.01, 0.01,
])
inv_w = 1.0 / (freq_weights + 0.01)
freq_weights_norm = inv_w / inv_w.sum()

def collate_fn(batch):
    t1_list, flair_list, t2_list = [], [], []
    h1_list, hf_list, h2_list = [], [], []
    reports_list = []
    for b in batch:
        t1_list.append(b["t1"]); flair_list.append(b["flair"]); t2_list.append(b["t2"])
        h1_list.append(b["has_t1"]); hf_list.append(b["has_flair"]); h2_list.append(b["has_t2"])
        reports_list.append(b["report"])
    return {"t1": torch.stack(t1_list), "flair": torch.stack(flair_list), "t2": torch.stack(t2_list),
            "has_t1": torch.stack(h1_list), "has_flair": torch.stack(hf_list), "has_t2": torch.stack(h2_list),
            "reports": reports_list}

loader.collate_fn = collate_fn
opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=1e-4)

for step, batch in enumerate(loader):
    if step > 20: break
    enc.train()
    with torch.no_grad():
        vt = enc(batch["t1"].to(dev), batch["flair"].to(dev), batch["t2"].to(dev),
                 batch["has_t1"], batch["has_flair"], batch["has_t2"])
    tids = tok(batch["reports"], return_tensors="pt", padding=True, truncation=True, max_length=256)["input_ids"].to(dev)
    te = emb(tids).to(dev).float()
    pad_mask = (tids == tok.pad_token_id).to(dev)

    # Check cross-attn directly
    q = enc.disease_proj.queries.expand(4, -1, -1)
    out, attn_w = enc.disease_proj.cross_attn(q, te, te, key_padding_mask=pad_mask)
    print(f"Step {step}: te nan={torch.isnan(te).any().item()} "
          f"cross_out nan={torch.isnan(out).any().item()} "
          f"q nan={torch.isnan(q).any().item()} "
          f"out range=[{out.min().item():.3f}, {out.max().item():.3f}] "
          f"queries grad_fn={q.grad_fn}")

    te_37 = enc.disease_proj.norm(out + q)
    print(f"  norm+res nan={torch.isnan(te_37).any().item()}")

    w = freq_weights_norm.to(dev).view(1, 37, 1)
    diff = vt - te_37
    se = diff.pow(2)
    loss = (se * w).sum() / se.numel()
    print(f"  loss={loss.item():.4f} loss.nan={torch.isnan(loss).any().item()}")

    opt.zero_grad()
    loss.backward()
    for name, p in enc.disease_proj.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  NAN grad in disease_proj.{name}!")
            break
    opt.step()
    print(f"  step done")
