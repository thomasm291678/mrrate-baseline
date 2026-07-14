"""Phase 3 evaluation: load checkpoint, generate reports on val, run evaluation_v2"""
import sys, json, torch
from pathlib import Path

sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")
sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet/evaluation/v2")

from encoder_v5 import ReportingModelV5, apply_optimizations
from mrrate_dataset import MRRateDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from evaluation_v2 import evaluate as eval_v2

apply_optimizations()
dev = torch.device("cuda")

LOG_DIR = Path("/home/jiaqigu/mrrate_hidnet/outputs/report_gen")
C3 = LOG_DIR / "phase3_latest.pt"
QWEN = "/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct"
DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"
G = 2
N_TOKENS = 3 * 5 * (G**3)

print(f"Loading encoder from phase3 checkpoint: {C3}")
enc = ReportingModelV5(llm_dim=2048, grid=G, base_ch=32).to(dev)
ckpt = torch.load(C3, map_location=dev, weights_only=False)
enc.load_state_dict(ckpt["encoder_state"], strict=False)
enc.eval()
print(f"Encoder loaded. params: {sum(p.numel() for p in enc.parameters()):,}")

tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    QWEN, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM")
llm = get_peft_model(llm, lora_cfg).to(dev)
if ckpt.get("llm_state"):
    llm.load_state_dict(ckpt["llm_state"], strict=False)
    print(f"LoRA weights restored")
llm.eval()

val_ds = MRRateDataset(DATA, "val", augment=False, batch_filter="batch27")
print(f"Val: {len(val_ds)} samples")

def collate(batch):
    t1_list, flair_list, t2_list = [], [], []
    h1_list, hf_list, h2_list = [], [], []
    reports_list = []
    for b in batch:
        t1_list.append(b["t1"]); flair_list.append(b["flair"]); t2_list.append(b["t2"])
        h1_list.append(b["has_t1"]); hf_list.append(b["has_flair"]); h2_list.append(b["has_t2"])
        reports_list.append(b["report"])
    return {
        "t1": torch.stack(t1_list), "flair": torch.stack(flair_list),
        "t2": torch.stack(t2_list),
        "has_t1": torch.stack(h1_list), "has_flair": torch.stack(hf_list),
        "has_t2": torch.stack(h2_list), "reports": reports_list,
    }

loader = torch.utils.data.DataLoader(
    val_ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=4,
    persistent_workers=True, prefetch_factor=2)

preds_out = []
refs_out = []

with torch.no_grad():
    for _, batch in enumerate(loader):
        B = batch["t1"].shape[0]
        vt = enc(batch["t1"].to(dev), batch["flair"].to(dev),
                 batch["t2"].to(dev), batch["has_t1"],
                 batch["has_flair"], batch["has_t2"])

        prefix_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False,
                                 return_tensors="pt").expand(B, -1).to(dev)
        prefix_embeds = llm.get_input_embeddings()(prefix_ids)
        combined = torch.cat([vt.to(prefix_embeds.dtype), prefix_embeds], dim=1)

        generated = llm.generate(
            inputs_embeds=combined, max_new_tokens=256, do_sample=False,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

        for i in range(B):
            gen_ids = generated[i]
            pt_len = N_TOKENS + prefix_ids.shape[1]
            out_ids = gen_ids[pt_len:]
            pred = tok.decode(out_ids, skip_special_tokens=True).strip()
            preds_out.append({"report": pred})
            refs_out.append({"report": batch["reports"][i]})

PREDS_PATH = LOG_DIR / "phase3_val_preds.json"
REFS_PATH = LOG_DIR / "phase3_val_refs.json"
json.dump(preds_out, open(PREDS_PATH, "w"), indent=2, ensure_ascii=False)
json.dump(refs_out, open(REFS_PATH, "w"), indent=2, ensure_ascii=False)
print(f"\nSaved {len(preds_out)} predictions -> {PREDS_PATH}")

# First 3 samples
for i, (p, g) in enumerate(zip(preds_out[:3], refs_out[:3])):
    print(f"\n--- Sample {i+1} ---")
    print(f"PRED: {p['report'][:300]}")
    print(f"GT:   {g['report'][:300]}")
