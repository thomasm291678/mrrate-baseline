import argparse, sys, time, signal
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoder_v5 import ReportingModelV5, apply_optimizations
from mrrate_dataset import MRRateDataset

# Magic: matmul precision decides float32 vs tf32 for all einsum/matmul,
# high = tf32 where safe, almost 2× speed with near-zero accuracy loss
apply_optimizations()


def contrastive_loss_fn(embeddings, patient_ids, temperature=0.07):
    if embeddings.shape[0] < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    sim = embeddings @ embeddings.T / temperature
    diag = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))
    pos_mask = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1))
    pos_mask = pos_mask.masked_fill(diag, False)
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    pos_sim = sim.masked_fill(~pos_mask, float("-inf")).logsumexp(dim=1, keepdim=True)
    all_sim = torch.logsumexp(sim, dim=1, keepdim=True)
    per_row = -(pos_sim - all_sim)
    loss = per_row[has_pos].mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return loss


def forward_contrastive(enc, batch, dev):
    mods = ["t1", "t2", "flair"]
    embs, pids = [], []
    for mod in mods:
        vol = batch[mod].to(dev)
        has = batch[f"has_{mod}"]
        if not has.any():
            continue
        idx = has.nonzero(as_tuple=True)[0]
        tok = enc.encode_raw(mod, vol[idx])
        emb = enc.contra_head(tok)
        embs.append(emb)
        pids.append(batch["patient_id"][idx].to(dev))
    if len(embs) == 0:
        return torch.tensor(0.0, device=dev, requires_grad=True)
    all_emb = torch.cat(embs, dim=0)
    all_pid = torch.cat(pids, dim=0)
    return contrastive_loss_fn(all_emb, all_pid, temperature=0.07)


def collate_fn(batch):
    t1_list, flair_list, t2_list = [], [], []
    h1_list, hf_list, h2_list = [], [], []
    reports_list = []
    pid_strs = []
    for b in batch:
        t1_list.append(b["t1"])
        flair_list.append(b["flair"])
        t2_list.append(b["t2"])
        h1_list.append(b["has_t1"])
        hf_list.append(b["has_flair"])
        h2_list.append(b["has_t2"])
        reports_list.append(b["report"])
        pid_strs.append(b.get("patient_uid", b.get("study_uid", "0")))
    unique_pids = {s: i for i, s in enumerate(sorted(set(pid_strs)))}
    pid_ids = torch.tensor([unique_pids[s] for s in pid_strs], dtype=torch.long)
    return {
        "t1": torch.stack(t1_list),
        "flair": torch.stack(flair_list),
        "t2": torch.stack(t2_list),
        "has_t1": torch.stack(h1_list),
        "has_flair": torch.stack(hf_list),
        "has_t2": torch.stack(h2_list),
        "reports": reports_list,
        "patient_id": pid_ids,
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
    log_path = log_dir / f"train_phase1_encoder_{ts}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"Phase 1: Encoder Contrastive Learning")
    log(f"Log: {log_path}")
    log(f"Grid: {args.grid}  BaseCh: {args.base_ch}  BatchID: {args.batch_id}")

    G = args.grid
    n_tokens_per_mod = 5 * (G ** 3)

    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=G, base_ch=args.base_ch).to(dev)
    for key in ["t1_proj", "t2_proj", "flair_proj"]:
        for p in getattr(enc, key).parameters():
            p.requires_grad = False

    total = sum(p.numel() for p in enc.parameters())
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    log(f"Encoder params: {total:,}  Trainable: {trainable:,}")
    log(f"Tokens per modality: {n_tokens_per_mod}")

    # torch.compile the encoder with reduce-overhead (CUDA graph replay for frozen parts)
    try:
        enc = torch.compile(enc, mode="reduce-overhead", dynamic=False)
        log("Encoder compiled (reduce-overhead mode)")
    except Exception as e:
        log(f"torch.compile skipped: {e}")

    # Fused AdamW: foreach=True for vectorized step, fused=True merges compute into 1 kernel
    # Magic: fused requires CUDA 12+ toolkit, falls back gracefully if unavailable
    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd)

    signal.signal(signal.SIGTERM, lambda *a: (sys.exit(0)))
    signal.signal(signal.SIGSEGV, lambda *a: (sys.exit(0)))

    train_ds = MRRateDataset(args.data_root, "train", augment=args.augment, batch_filter=args.batch_id)
    val_ds = MRRateDataset(args.data_root, "val", augment=False, batch_filter=args.batch_id)
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}  Workers: {args.num_workers}")

    # Magic numbers: 8 workers + prefetch_factor=4 saturate NAS I/O on 16-core machines
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
    log(f"Total opt steps: {total_steps}  LR: {args.lr}  Fused AdamW: yes")

    start_epoch, global_step, best_loss = 1, 0, float("inf")

    if args.auto_resume:
        auto_path = log_dir / "phase1_latest.pt"
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
                loss = forward_contrastive(enc, batch, dev) / args.ga_steps
                if loss.requires_grad:
                    loss.backward()

                epoch_loss += loss.item()
                ga_count += 1

                if ga_count % args.ga_steps == 0:
                    global_step += 1
                    nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], args.grad_clip)
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
                                        best_loss, log_dir / "phase1_latest.pt")
                        log_f.flush()

                    if global_step % args.save_interval == 0:
                        save_checkpoint(enc, opt, sched, epoch, global_step,
                                        best_loss,
                                        log_dir / f"phase1_step{global_step}.pt")
                        log(f"Saved: step {global_step}")

            except Exception as e:
                log(f"ERROR at step {step}: {e}")
                import traceback; traceback.print_exc()
                save_checkpoint(enc, opt, sched, epoch, global_step,
                                best_loss, log_dir / "phase1_latest.pt")
                log_f.flush()
                raise

        if ga_count > 0:
            global_step += 1
            nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

        log(f"Epoch {epoch} done. avg_loss: {epoch_loss/max(1,step+1):.4f}")

    log("Phase 1 finished.")
    log_f.flush()
    log_f.close()


def main():
    p = argparse.ArgumentParser("Phase 1: Encoder Contrastive Learning")
    p.add_argument("--data_root", type=str, default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--log_dir", type=str, default="outputs/report_gen")
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ga_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)  # magic: 16-core sweet spot
    p.add_argument("--prefetch_factor", type=int, default=4)  # magic: 4x pipeline depth
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--auto_save_interval", type=int, default=50)
    p.add_argument("--batch_id", type=str, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
