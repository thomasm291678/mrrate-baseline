import argparse, sys, time, signal, json, os
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_v5 import ReportingModelV5, apply_optimizations
from mrrate_dataset import MRRateDataset

apply_optimizations()

APPROX_DISEASE_FREQ = [
    0.30, 0.05, 0.25, 0.03, 0.02, 0.01, 0.02,
    0.05, 0.08, 0.02, 0.10, 0.03, 0.005,
    0.20, 0.08, 0.02, 0.15, 0.08, 0.04,
    0.06, 0.05, 0.10, 0.02, 0.04,
    0.03, 0.02, 0.04, 0.01, 0.01, 0.03,
    0.40, 0.15, 0.05, 0.06, 0.02, 0.01, 0.01,
]


def alignment_loss_fn(vt, text_embeds, enc, freq_weights):
    """
    Symmetric disease-aware alignment via shared cross-attention.

    Visual: [B, 120, 2048] → DiseaseAwareProjector → [B, 37, 2048]
    Text:   [B, T, 2048]  → DiseaseAwareProjector → [B, 37, 2048]
    Loss:   per-disease MSE weighted by inverse frequency.
    """
    te = text_embeds.float()
    if te.shape[1] == 0:
        return torch.tensor(0.0, device=vt.device, requires_grad=True)

    te_37 = enc.disease_proj(te)
    w = freq_weights.to(vt.device).float().view(1, 37, 1)
    return F.mse_loss(vt.float() * w, te_37 * w)


def collate_fn(batch):
    t1_list, flair_list, t2_list = [], [], []
    h1_list, hf_list, h2_list = [], [], []
    reports_list = []
    for b in batch:
        t1_list.append(b["t1"])
        flair_list.append(b["flair"])
        t2_list.append(b["t2"])
        h1_list.append(b["has_t1"])
        hf_list.append(b["has_flair"])
        h2_list.append(b["has_t2"])
        reports_list.append(b["report"])
    return {
        "t1": torch.stack(t1_list),
        "flair": torch.stack(flair_list),
        "t2": torch.stack(t2_list),
        "has_t1": torch.stack(h1_list),
        "has_flair": torch.stack(hf_list),
        "has_t2": torch.stack(h2_list),
        "reports": reports_list,
    }


def save_checkpoint(enc, opt, sched, epoch, global_step, best_loss, path):
    state = {
        "encoder_state": enc.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
    }
    torch.save(state, path)


def load_checkpoint(path, enc, opt, sched, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    enc.load_state_dict(ckpt["encoder_state"], strict=False)
    try:
        opt.load_state_dict(ckpt["optimizer_state"])
    except ValueError:
        print(f"  [WARN] Skipping optimizer state (param group mismatch)")
    if sched and ckpt.get("scheduler_state"):
        sched.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_loss"]


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_phase2_projector_{ts}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log("Phase 2: Disease-Aware Projector Alignment (37 queries + freq-weighted)")
    log(f"Log: {log_path}")
    log(f"Encoder checkpoint: {args.encoder_ckpt}")

    if args.freq_weights:
        freq_weights = torch.tensor([float(x) for x in args.freq_weights.split(",")])
    else:
        freq_weights = torch.tensor(APPROX_DISEASE_FREQ)
    inv_w = 1.0 / (freq_weights + 0.01)
    freq_weights_norm = inv_w / inv_w.sum()
    log(f"Freq weights (normalized inv-freq): {freq_weights_norm.tolist()}")
    log(f"  Top-5 heaviest diseases: {freq_weights_norm.topk(5).indices.tolist()}")

    from transformers import AutoTokenizer

    G = args.grid

    enc = ReportingModelV5(
        llm_dim=args.llm_dim, grid=G, base_ch=args.base_ch,
        n_vt_out=1).to(dev)
    log(f"Visual token output: {enc.n_tokens_out} (37 disease-aware tokens)")

    if args.encoder_ckpt:
        log("Loading Phase 1 encoder weights...")
        phase1 = torch.load(args.encoder_ckpt, map_location=dev, weights_only=False)
        enc.load_state_dict(phase1["encoder_state"], strict=False)
        log(f"  Loaded Phase 1 encoder (epoch {phase1.get('epoch','?')}, step {phase1.get('global_step','?')})")

    for p in enc.t1_enc.parameters():
        p.requires_grad = False
    for p in enc.t2_enc.parameters():
        p.requires_grad = False
    for p in enc.flair_enc.parameters():
        p.requires_grad = False
    log("Froze MRRCNN encoders")

    total = sum(p.numel() for p in enc.parameters())
    trainable_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    log(f"Encoder params: {total:,}  Trainable: {trainable_enc:,}")

    tok = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from safetensors.torch import load_file as st_load
    idx = json.load(open(os.path.join(args.qwen_path, "model.safetensors.index.json")))
    emb_path = os.path.join(args.qwen_path, idx["weight_map"]["model.embed_tokens.weight"])
    emb = nn.Embedding.from_pretrained(st_load(emb_path)["model.embed_tokens.weight"].to(torch.bfloat16))
    emb.weight.requires_grad = False
    log("Qwen embedding loaded directly (no transformer, ~600MB)")

    trainable = [p for p in enc.parameters() if p.requires_grad]
    log(f"Total trainable: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    signal.signal(signal.SIGTERM, lambda *a: (sys.exit(0)))
    signal.signal(signal.SIGSEGV, lambda *a: (sys.exit(0)))

    train_ds = MRRateDataset(args.data_root, "train", augment=False, batch_filter=args.batch_id)
    val_ds = MRRateDataset(args.data_root, "val", augment=False, batch_filter=args.batch_id)
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}  Workers: {args.num_workers}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    total_steps = args.epochs * (len(loader) // args.ga_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps + 10, eta_min=args.lr * 0.01,
    )
    log(f"Total opt steps: {total_steps}  LR: {args.lr}")

    start_epoch, global_step, best_loss = 1, 0, float("inf")

    if args.auto_resume:
        auto_path = log_dir / "phase2_latest.pt"
        if auto_path.exists():
            log(f"Auto-resume from {auto_path}")
            start_epoch, global_step, best_loss = load_checkpoint(
                auto_path, enc, opt, sched, dev)
            start_epoch += 1
            sched.T_max = total_steps + 10
            log(f"Resumed: epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        enc.train()
        epoch_loss = 0.0
        ga_count = 0
        opt.zero_grad(set_to_none=True)

        t0 = time.time()
        for step, batch in enumerate(loader):
            try:
                vt = enc(batch["t1"].to(dev), batch["flair"].to(dev),
                         batch["t2"].to(dev), batch["has_t1"],
                         batch["has_flair"], batch["has_t2"])

                target_ids = tok(batch["reports"], return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=args.max_text_len)["input_ids"]
                text_embeds = emb(target_ids).to(dev)

                loss = alignment_loss_fn(vt, text_embeds, enc, freq_weights_norm)
                loss = loss / args.ga_steps
                if loss.requires_grad:
                    loss.backward()

                epoch_loss += loss.item()
                ga_count += 1

                if ga_count % args.ga_steps == 0:
                    global_step += 1
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    opt.step()
                    if sched:
                        sched.step()
                    opt.zero_grad(set_to_none=True)
                    ga_count = 0

                    if global_step % args.log_interval == 0:
                        n_steps = global_step
                        elapsed = time.time() - t0
                        eta_s = (total_steps - n_steps) * elapsed / max(n_steps, 1)
                        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        log(f"[E{epoch:03d} {n_steps/total_steps*100:5.1f}% S{n_steps:05d}] "
                            f"loss={loss.item()*args.ga_steps:.4f} avg_loss={epoch_loss/max(1,step+1):.4f} "
                            f"lr={opt.param_groups[0]['lr']:.1e} mem={mem:.1f}GB eta={eta_s/3600:.1f}h")

                    if global_step % args.auto_save_interval == 0:
                        save_checkpoint(enc, opt, sched, epoch, global_step,
                                        best_loss, log_dir / "phase2_latest.pt")
                        log_f.flush()

                    if global_step % args.save_interval == 0:
                        save_checkpoint(enc, opt, sched, epoch, global_step,
                                        best_loss,
                                        log_dir / f"phase2_step{global_step}.pt")
                        log(f"Saved: step {global_step}")

            except Exception as e:
                log(f"ERROR at step {step}: {e}")
                import traceback; traceback.print_exc()
                save_checkpoint(enc, opt, sched, epoch, global_step,
                                best_loss, log_dir / "phase2_latest.pt")
                log_f.flush()
                raise

        if ga_count > 0:
            global_step += 1
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        log(f"Epoch {epoch} done. avg_loss: {epoch_loss/max(1,step+1):.4f}")

    log("Phase 2 finished.")
    log_f.flush()
    log_f.close()


def main():
    p = argparse.ArgumentParser("Phase 2: Disease-Aware Projector Alignment")
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--qwen_path", type=str, default="/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct")
    p.add_argument("--log_dir", type=str, default="outputs/report_gen")
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--freq_weights", type=str, default=None,
                   help="comma-separated 37 disease frequencies, e.g. '0.3,0.05,0.25,...'")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ga_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_text_len", type=int, default=256)
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--auto_save_interval", type=int, default=50)
    p.add_argument("--batch_id", type=str, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
