import paramiko, os, time

HOST = "10.176.60.70"; USER = "jiaqigu"; PASS = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet"
LOCAL = r"C:\Users\HP\Documents\5555"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

# ====== 1. Install deps via tmux (long-running) ======
print("Starting pip install in tmux 'dep' on farm02...")
shell = "pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && pip3 install --no-cache-dir transformers peft accelerate scipy nibabel pandas tqdm && touch /tmp/deps_done"
c.exec_command(f'tmux kill-session -t dep 2>/dev/null; tmux new-session -d -s dep "{shell}"')

# ====== 2. Upload files while deps install ======
print("Uploading files...")
sftp = c.open_sftp()

files_uploaded = []
sftp.put(os.path.join(LOCAL, "encoder_v5.py"), f"{REMOTE}/encoder_v5.py"); files_uploaded.append("encoder_v5.py")
sftp.put(os.path.join(LOCAL, "encoder_v4.py"), f"{REMOTE}/encoder_v4.py"); files_uploaded.append("encoder_v4.py")
sftp.put(os.path.join(LOCAL, "server_code", "mrrate_dataset.py"), f"{REMOTE}/mrrate_dataset.py"); files_uploaded.append("mrrate_dataset.py")

# train_v5.py for Phase 1: use train_v5_phase1.py renamed
with open(os.path.join(LOCAL, "train_v5_phase1.py")) as f:
    code = f.read()
with sftp.open(f"{REMOTE}/train_v5.py", "w") as f:
    f.write(code)
files_uploaded.append("train_v5.py (from train_v5_phase1.py)")

# generate_init.py: create from scratch
generate_init_script = '''import argparse, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_v5 import ReportingModelV5

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=args.grid, base_ch=args.base_ch)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    state = {"encoder_state": enc.state_dict(), "seed": args.seed, "grid": args.grid, "base_ch": args.base_ch}
    torch.save(state, args.out)
    print(f"Initial weights saved to {args.out}")

if __name__ == "__main__":
    main()
'''
with sftp.open(f"{REMOTE}/generate_init.py", "w") as f:
    f.write(generate_init_script)
files_uploaded.append("generate_init.py")

# merge_encoders.py: create from scratch
merge_encoders_script = '''import argparse, torch, torch.nn.functional as F, numpy as np, sys, itertools, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_v5 import ReportingModelV5
from mrrate_dataset import MRRateDataset

def _reorder_mha_weight(w, perm): return w[perm].contiguous()
def _reorder_mha_bias(b, perm): return b[perm].contiguous() if b is not None else None

def align_stage_attention(state_dicts, enc, dev="cpu"):
    mod_weights = {}
    for mod in ["t1_enc", "flair_enc", "t2_enc"]:
        for stage_id in range(5):
            key_bias = f"{mod}.stages.{stage_id}.block1.attn.out_proj.bias"
            if key_bias not in state_dicts[0]: continue
            weights = [sd[key_bias].to(dev) for sd in state_dicts]
            avg = torch.stack(weights).mean(dim=0)
            best_perm = None; best_score = -1
            for ref_idx in range(len(state_dicts)):
                for perm in itertools.permutations(range(weights[ref_idx].size(0))):
                    perm = torch.tensor(perm, device=dev)
                    permuted = weights[ref_idx][perm]
                    score = F.cosine_similarity(permuted.unsqueeze(0), avg.unsqueeze(0), dim=-1).sum().item()
                    if score > best_score:
                        best_score = score; best_perm = perm
            mod_weights[(mod, stage_id)] = best_perm.cpu()
            print(f"  Aligned {mod} stage{stage_id}: perm={best_perm.tolist()}")
    return mod_weights

def fedavg(state_dicts, align_perms, sample_counts, enc, dev="cpu"):
    total = sum(sample_counts)
    merged = {}
    skip_prefixes = ["t1_enc.stem", "t1_enc.stages.0.block0", "t1_enc.stages.1.block0",
                     "t1_enc.stages.2.block0", "t1_enc.stages.3.block0", "t1_enc.stages.4.block0",
                     "t2_enc.stem", "t2_enc.stages.0.block0", "t2_enc.stages.1.block0",
                     "t2_enc.stages.2.block0", "t2_enc.stages.3.block0", "t2_enc.stages.4.block0",
                     "flair_enc.stem", "flair_enc.stages.0.block0", "flair_enc.stages.1.block0",
                     "flair_enc.stages.2.block0", "flair_enc.stages.3.block0", "flair_enc.stages.4.block0"]

    for k in state_dicts[0].keys():
        if k.startswith("cnn_dim") or k.startswith("grid") or k.startswith("n_tokens"): continue
        is_bn_running = ".running_mean" in k or ".running_var" in k
        is_contrastive = "contrastive_head" in k or "ContrasiveHead" in k
        is_attn = ".attn." in k
        is_mlp = ".mlp." in k

        if is_bn_running or is_contrastive:
            continue

        tensors = [sd[k].float().to(dev) for sd in state_dicts]
        if is_attn:
            for mod in ["t1_enc", "flair_enc", "t2_enc"]:
                if k.startswith(mod):
                    for stage_id in range(5):
                        stage_prefix = f"{mod}.stages.{stage_id}.block1.attn."
                        if stage_prefix in k:
                            perm = align_perms.get((mod, stage_id))
                            if perm is not None:
                                if "in_proj" in k:
                                    tensors = [sd[k].float().to(dev) for sd in state_dicts]
                                elif "out_proj" in k:
                                    tensors = [sd[k].float().to(dev) for sd in state_dicts]

        weighted = torch.zeros_like(tensors[0])
        for t, n in zip(tensors, sample_counts):
            weighted += t * (n / total)
        merged[k] = weighted

    return merged

def recalibrate_bn(enc, data_root, limit=20000, batch_size=8, dev="cuda"):
    import random
    ds = MRRateDataset(data_root, "train", augment=False)
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    if limit and limit < len(all_indices):
        all_indices = all_indices[:limit]

    enc.train()
    enc.to(dev)
    with torch.no_grad():
        for start in range(0, len(all_indices), batch_size):
            batch_idx = all_indices[start:start + batch_size]
            t1_list, flair_list, t2_list = [], [], []
            h1, hf, h2 = [], [], []
            for idx in batch_idx:
                item = ds[idx]
                t1_list.append(item["t1"]); flair_list.append(item["flair"]); t2_list.append(item["t2"])
                h1.append(item["has_t1"]); hf.append(item["has_flair"]); h2.append(item["has_t2"])
            if not t1_list: continue
            t1 = torch.stack(t1_list).to(dev); flair = torch.stack(flair_list).to(dev); t2 = torch.stack(t2_list).to(dev)
            ht1 = torch.stack(h1); hfl = torch.stack(hf); ht2 = torch.stack(h2)
            _ = enc(t1, flair, t2, ht1, hfl, ht2)
    enc.eval()
    return enc

def main():
    p = argparse.ArgumentParser("FedAvg MRRCNN Merger")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--sample_counts", nargs="+", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--data_root", type=str, default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--recal_limit", type=int, default=20000)
    p.add_argument("--recal_batch_size", type=int, default=8)
    p.add_argument("--skip_recalibrate", action="store_true")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}, CKPTs: {len(args.ckpts)}, Samples: {args.sample_counts}")

    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=args.grid, base_ch=args.base_ch)
    state_dicts = []
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("encoder_state", ckpt)
        state_dicts.append(sd)
        print(f"  Loaded: {path}")

    print("\n=== Step 1: Head Alignment ===")
    align_perms = align_stage_attention(state_dicts, enc, dev)

    print("\n=== Step 2: FedAvg Weighted Merge ===")
    merged = fedavg(state_dicts, align_perms, args.sample_counts, enc, dev)
    enc.load_state_dict(merged, strict=False)
    print(f"  Merged {len(merged)} parameters")

    if not args.skip_recalibrate:
        print(f"\n=== Step 3: BN Recalibration ({args.recal_limit} samples) ===")
        enc = recalibrate_bn(enc, args.data_root, args.recal_limit, args.recal_batch_size, dev)
        print("  BN recalibrated")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    state = {"encoder_state": enc.state_dict()}
    torch.save(state, args.out)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
'''
with sftp.open(f"{REMOTE}/merge_encoders.py", "w") as f:
    f.write(merge_encoders_script)
files_uploaded.append("merge_encoders.py")

sftp.close()
print(f"Uploaded: {', '.join(files_uploaded)}")

# ====== 3. Wait for deps to finish ======
print("\nWaiting for pip install to finish...")
for i in range(20):
    time.sleep(30)
    _, o, _ = c.exec_command("test -f /tmp/deps_done && echo YES || echo NO")
    if "YES" in o.read().decode():
        print("Deps installed!")
        break
    print(f"  Still waiting... ({i+1}/20)")

# Verify
_, o, _ = c.exec_command("python3 -c 'import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.cuda.is_available())' 2>&1")
print(o.read().decode().strip())

c.close()
print("\nDeploy done.")
