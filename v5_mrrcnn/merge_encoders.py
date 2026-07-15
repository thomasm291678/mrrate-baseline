import torch
import torch.nn as nn
import argparse
import itertools
import time
from pathlib import Path
from collections import OrderedDict

from encoder_v5 import ReportingModelV5, MRRCNN
from mrrate_dataset import MRRateDataset


def get_mha_keys(prefix):
    return [
        f"{prefix}stage_attn.attn.in_proj_weight",
        f"{prefix}stage_attn.attn.in_proj_bias",
        f"{prefix}stage_attn.attn.out_proj.weight",
        f"{prefix}stage_attn.attn.out_proj.bias",
    ]


def should_skip(key):
    if "contra_head" in key:
        return True
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return True
    return False


def find_head_permutation(ref_out_proj, model_out_proj, heads=4):
    dim = ref_out_proj.shape[0]
    d_head = dim // heads

    best_perm = None
    best_cost = float("inf")

    for perm in itertools.permutations(range(heads)):
        cost = 0.0
        for h_ref, h_model in enumerate(perm):
            col_ref = ref_out_proj[:, h_ref * d_head:(h_ref + 1) * d_head]
            col_model = model_out_proj[:, h_model * d_head:(h_model + 1) * d_head]
            cost += torch.norm(col_ref - col_model, p="fro").item() ** 2
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return best_perm


def apply_head_permutation(state_dict, prefix, perm, heads=4, dim=512):
    d_head = dim // heads

    in_proj_key = f"{prefix}stage_attn.attn.in_proj_weight"
    in_proj_bias_key = f"{prefix}stage_attn.attn.in_proj_bias"
    out_proj_key = f"{prefix}stage_attn.attn.out_proj.weight"
    out_proj_bias_key = f"{prefix}stage_attn.attn.out_proj.bias"

    in_proj_w = state_dict[in_proj_key]
    in_proj_b = state_dict[in_proj_bias_key]
    out_proj_w = state_dict[out_proj_key]
    out_proj_b = state_dict[out_proj_bias_key]

    new_in_proj_w = torch.zeros_like(in_proj_w)
    new_in_proj_b = torch.zeros_like(in_proj_b)
    new_out_proj_w = torch.zeros_like(out_proj_w)
    new_out_proj_b = torch.zeros_like(out_proj_b)

    for h_ref, h_model in enumerate(perm):
        src_start = h_model * d_head
        src_end = (h_model + 1) * d_head
        dst_start = h_ref * d_head
        dst_end = (h_ref + 1) * d_head

        new_in_proj_w[dst_start:dst_end, :] = in_proj_w[src_start:src_end, :]
        new_in_proj_w[dim + dst_start:dim + dst_end, :] = in_proj_w[dim + src_start:dim + src_end, :]
        new_in_proj_w[2 * dim + dst_start:2 * dim + dst_end, :] = in_proj_w[2 * dim + src_start:2 * dim + src_end, :]

        new_in_proj_b[dst_start:dst_end] = in_proj_b[src_start:src_end]
        new_in_proj_b[dim + dst_start:dim + dst_end] = in_proj_b[dim + src_start:dim + src_end]
        new_in_proj_b[2 * dim + dst_start:2 * dim + dst_end] = in_proj_b[2 * dim + src_start:2 * dim + src_end]

        new_out_proj_w[:, dst_start:dst_end] = out_proj_w[:, src_start:src_end]
        new_out_proj_b[dst_start:dst_end] = out_proj_b[src_start:src_end]

    state_dict[in_proj_key] = new_in_proj_w
    state_dict[in_proj_bias_key] = new_in_proj_b
    state_dict[out_proj_key] = new_out_proj_w
    state_dict[out_proj_bias_key] = new_out_proj_b


def align_all_attention_heads(all_states, heads=4, dim=512):
    d_head = dim // heads
    modalities = ["t1_enc", "t2_enc", "flair_enc"]
    ref_state = all_states[0]

    for mod in modalities:
        ref_out_key = f"{mod}.stage_attn.attn.out_proj.weight"
        ref_out_proj = ref_state[ref_out_key]

        print(f"  [{mod}] reference out_proj heads:")
        for h in range(heads):
            print(f"    head {h}: ||col|| = {ref_out_proj[:, h*d_head:(h+1)*d_head].norm():.2f}")

    for idx in range(1, len(all_states)):
        state = all_states[idx]
        for mod in modalities:
            ref_out_key = f"{mod}.stage_attn.attn.out_proj.weight"
            model_out_key = f"{mod}.stage_attn.attn.out_proj.weight"
            ref_out = ref_state[ref_out_key]
            model_out = state[model_out_key]

            perm = find_head_permutation(ref_out, model_out, heads=heads)
            print(f"  GPU{idx} {mod}: best permutation = {perm}")

            apply_head_permutation(state, f"{mod}.", perm, heads=heads, dim=dim)


def fedavg_merge(all_states, sample_counts):
    total = sum(sample_counts)
    merged = OrderedDict()

    all_keys = set(all_states[0].keys())
    for key in sorted(all_keys):
        if should_skip(key):
            continue

        merged[key] = sum(
            sample_counts[i] / total * all_states[i][key]
            for i in range(len(all_states))
        )

    return merged


def recalibrate_bn(merged_state, data_root, device, batch_size=4, num_workers=4, max_samples=None):
    print("\n[BN Recalibration] Loading full training data...")
    full_ds = MRRateDataset(data_root, "train", augment=False, batch_filter=None)
    if max_samples and max_samples < len(full_ds):
        indices = torch.randperm(len(full_ds))[:max_samples].tolist()
        full_ds = torch.utils.data.Subset(full_ds, indices)
    print(f"  Total samples: {len(full_ds)}")

    loader = torch.utils.data.DataLoader(
        full_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    enc = ReportingModelV5(llm_dim=2048, grid=2, base_ch=32).to(device)
    missing, unexpected = enc.load_state_dict(merged_state, strict=False)

    for m in enc.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.momentum = 0.1
            m.train()

    print(f"  Running forward pass to update BN stats...")
    n_total = len(loader)
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch["has_t1"].any():
                enc.t1_enc(batch["t1"][batch["has_t1"]].to(device))
            if batch["has_flair"].any():
                enc.flair_enc(batch["flair"][batch["has_flair"]].to(device))
            if batch["has_t2"].any():
                enc.t2_enc(batch["t2"][batch["has_t2"]].to(device))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_total - i - 1)
                print(f"    [{i+1}/{n_total}] eta={eta/60:.1f}min")

    elapsed_total = time.time() - t0
    print(f"  Done in {elapsed_total/60:.1f}min")

    final_state = OrderedDict()
    for key, val in enc.state_dict().items():
        if key in merged_state and not should_skip(key):
            final_state[key] = val
        elif "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            final_state[key] = val

    return final_state


def main():
    p = argparse.ArgumentParser("FedAvg merge for Phase 1 V5 encoders")
    p.add_argument("--ckpts", type=str, nargs="+", required=True,
                   help="Paths to the 4 GPU checkpoint .pt files")
    p.add_argument("--sample_counts", type=int, nargs="+", required=True,
                   help="Number of training samples per GPU (4 integers)")
    p.add_argument("--out", type=str, default="outputs/merged_encoder.pt")
    p.add_argument("--data_root", type=str, default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--skip_recalibrate", action="store_true",
                   help="Skip BN recalibration (not recommended)")
    p.add_argument("--recal_batch_size", type=int, default=4)
    p.add_argument("--recal_workers", type=int, default=4)
    p.add_argument("--recal_limit", type=int, default=20000,
                   help="Max samples for BN recalibration (default 20k, set 0 for all)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=512)
    args = p.parse_args()

    assert len(args.ckpts) == 4, "Need exactly 4 checkpoints"
    assert len(args.sample_counts) == 4, "Need exactly 4 sample counts"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Checkpoints: {args.ckpts}")
    print(f"Sample counts: {args.sample_counts}")

    print("\n[Step 1] Loading checkpoints...")
    all_states = []
    for i, path in enumerate(args.ckpts):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["encoder_state"]
        all_states.append(state)
        n_keys = len(state)
        bn_keys = sum(1 for k in state if "running" in k)
        mha_keys = sum(1 for k in state if "stage_attn.attn" in k)
        print(f"  GPU{i}: {n_keys} keys ({bn_keys} BN stats, {mha_keys} MHA)")

    print("\n[Step 2] Aligning StageAttention heads...")
    align_all_attention_heads(all_states, heads=args.heads, dim=args.dim)

    print("\n[Step 3] FedAvg weighted merge...")
    merged = fedavg_merge(all_states, args.sample_counts)
    n_merged = len(merged)
    n_skipped = len(all_states[0]) - n_merged
    print(f"  Merged: {n_merged} keys  Skipped: {n_skipped} keys (contra_head + BN stats)")

    if not args.skip_recalibrate:
        print("\n[Step 4] BN recalibration on full dataset...")
        merged = recalibrate_bn(
            merged, args.data_root, dev,
            batch_size=args.recal_batch_size,
            num_workers=args.recal_workers,
            max_samples=args.recal_limit if args.recal_limit > 0 else None,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": merged}, out_path)
    print(f"\nSaved merged weights to {out_path}")
    print(f"  Total keys: {len(merged)}")


if __name__ == "__main__":
    main()
