import argparse, sys, time, signal
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_v5 import ReportingModelV5, MRRCNN
from mrrate_dataset import MRRateDataset

IGNORE_INDEX = -100


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
    n_neg = (~pos_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
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
    for i, mod in enumerate(mods):
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


def build_embeddings(llm, tok, vt_float, target_ids, B, n_vt, device):
    text_embeds = llm.get_input_embeddings()(target_ids)
    combined = torch.cat([vt_float.to(text_embeds.dtype), text_embeds], dim=1)
    ignore_labels = torch.full((B, n_vt), IGNORE_INDEX, dtype=target_ids.dtype, device=device)
    labels = torch.cat([ignore_labels, target_ids], dim=1)
    attn_mask = torch.cat([
        torch.ones(B, n_vt, device=device, dtype=torch.long),
        (target_ids != tok.pad_token_id).long()], dim=1)
    return combined, labels, attn_mask


def alignment_loss_fn(vt, text_embeds, max_text_len=128):
    vt_p = vt.float().mean(dim=1)
    te = text_embeds[:, :max_text_len].float()
    if te.shape[1] == 0:
        return torch.tensor(0.0, device=vt.device, requires_grad=True)
    mask = (te.sum(dim=-1) != 0).float().unsqueeze(-1)
    te_m = (te * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.mse_loss(vt_p, te_m)


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


def save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step,
                    best_loss, metrics, path, step_loss=None):
    state = {
        "encoder_state": enc.state_dict(),
        "llm_state": llm.state_dict() if llm is not None else {},
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict() if sched else {},
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch, "global_step": global_step,
        "best_loss": best_loss, "metrics": metrics,
        "step_loss": step_loss,
    }
    torch.save(state, path)


def load_checkpoint(path, enc, llm, opt, sched, scaler, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    enc.load_state_dict(ckpt["encoder_state"], strict=False)
    if llm is not None and ckpt.get("llm_state"):
        llm.load_state_dict(ckpt["llm_state"], strict=False)
    try:
        opt.load_state_dict(ckpt["optimizer_state"])
    except ValueError:
        print(f"  [WARN] Skipping optimizer state (param group mismatch)")
    if sched and ckpt.get("scheduler_state"):
        sched.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_loss"], ckpt.get("metrics", [])


def train(args):
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_v5_{args.modality}_{args.phase}_{ts}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"V5 Log: {log_path}")
    log(f"Phase: {args.phase}  Modality: {args.modality}  Grid: {args.grid}")

    is_single = args.modality != "all"
    G = args.grid
    n_tokens_per_mod = 5 * (G ** 3)
    n_vt = n_tokens_per_mod if is_single else 3 * n_tokens_per_mod

    enc = ReportingModelV5(llm_dim=args.llm_dim, grid=G, base_ch=args.base_ch).to(dev)
    log(f"Encoder params: {sum(p.numel() for p in enc.parameters()):,}")
    log(f"  tokens/mod: {n_tokens_per_mod}  total: {n_vt}")

    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=dev, weights_only=False)
        enc.load_state_dict(init_ckpt["encoder_state"], strict=True)
        log(f"Loaded shared init weights from {args.init_from} (seed={init_ckpt.get('seed','?')})")

    if args.phase == "encoder":
        for key in ["t1_proj", "t2_proj", "flair_proj"]:
            for p in getattr(enc, key).parameters():
                p.requires_grad = False
        log("Froze projector heads")

    if args.phase in ("encoder",):
        llm = None
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            args.qwen_path, torch_dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=True)
        llm.resize_token_embeddings(len(tok))

        if args.phase == "projector":
            for p in llm.parameters():
                p.requires_grad = False
            log("Qwen: embedding only (no LoRA, no forward)")
        elif args.phase == "qwen":
            from peft import LoraConfig, get_peft_model
            llm.gradient_checkpointing_enable()
            llm.config.use_cache = False
            lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                  lora_dropout=args.lora_drop,
                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                  bias="none", task_type="CAUSAL_LM")
            llm = get_peft_model(llm, lora_cfg).to(dev)
            lo = sum(p.numel() for p in llm.parameters() if p.requires_grad)
            log(f"Qwen LoRA params: {lo:,}")

        if args.phase == "projector":
            for p in enc.t1_enc.parameters():
                p.requires_grad = False
            for p in enc.t2_enc.parameters():
                p.requires_grad = False
            for p in enc.flair_enc.parameters():
                p.requires_grad = False
            log("Froze V5 encoder (training projector only)")

        if args.phase == "qwen":
            for p in enc.parameters():
                p.requires_grad = False
            log("Froze V5+projector (training Qwen LoRA only)")

    trainable = [p for p in enc.parameters() if p.requires_grad]
    if llm is not None:
        trainable += [p for p in llm.parameters() if p.requires_grad]
    log(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    signal.signal(signal.SIGTERM, lambda *a: (sys.exit(0)))
    signal.signal(signal.SIGSEGV, lambda *a: (sys.exit(0)))

    train_ds = MRRateDataset(args.data_root, "train", augment=args.augment, batch_filter=args.batch_id)
    val_ds = MRRateDataset(args.data_root, "val", augment=False, batch_filter=args.batch_id)
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}  Workers: {args.num_workers}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    total_steps = args.epochs * (len(loader) // args.ga_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps + 10, eta_min=args.lr * 0.01,
    )
    log(f"Total opt steps: {total_steps}")

    start_epoch, global_step, best_loss = 1, 0, float("inf")
    metrics_history = []

    if args.auto_resume:
        auto_path = log_dir / "latest_step.pt"
        if auto_path.exists():
            log(f"Auto-resume from {auto_path}")
            start_epoch, global_step, best_loss, metrics_history = load_checkpoint(
                auto_path, enc, llm, opt, sched, scaler, dev)
            start_epoch += 1
            sched.T_max = total_steps + 10
            log(f"Resumed: epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        enc.train()
        if llm is not None:
            llm.train()
        epoch_loss = 0.0
        ga_count = 0
        opt.zero_grad()

        t0 = time.time()
        for step, batch in enumerate(loader):
            try:
                B = batch["t1"].shape[0]

                if args.phase == "encoder":
                    loss = forward_contrastive(enc, batch, dev) / args.ga_steps
                elif args.phase == "projector":
                    vt = enc(batch["t1"].to(dev), batch["flair"].to(dev),
                             batch["t2"].to(dev), batch["has_t1"],
                             batch["has_flair"], batch["has_t2"])
                    target_ids = tok(batch["reports"], return_tensors="pt",
                                     padding=True, truncation=True, max_length=args.max_text_len)["input_ids"].to(dev)
                    text_embeds = llm.get_input_embeddings()(target_ids)
                    loss = alignment_loss_fn(vt, text_embeds) / args.ga_steps
                elif args.phase == "qwen":
                    vt = enc(batch["t1"].to(dev), batch["flair"].to(dev),
                             batch["t2"].to(dev), batch["has_t1"],
                             batch["has_flair"], batch["has_t2"])
                    target_ids = tok(batch["reports"], return_tensors="pt",
                                     padding=True, truncation=True, max_length=args.max_text_len)["input_ids"].to(dev)
                    embeds, labels, am = build_embeddings(llm, tok, vt, target_ids, B, n_vt, dev)
                    if scaler:
                        with torch.amp.autocast("cuda"):
                            loss = llm(inputs_embeds=embeds, attention_mask=am, labels=labels).loss / args.ga_steps
                        scaler.scale(loss).backward()
                    else:
                        loss = llm(inputs_embeds=embeds, attention_mask=am, labels=labels).loss / args.ga_steps
                        loss.backward()

                if args.phase != "qwen" or not scaler:
                    if loss.requires_grad:
                        loss.backward()

                epoch_loss += loss.item()
                ga_count += 1

                if ga_count % args.ga_steps == 0:
                    global_step += 1
                    if scaler:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                        opt.step()
                    if sched:
                        sched.step()
                    opt.zero_grad()
                    ga_count = 0

                    if global_step % args.log_interval == 0:
                        avg_loss = epoch_loss / (global_step - (epoch - 1) * total_steps // args.epochs)
                        avg_loss = epoch_loss / max(1, global_step - (start_epoch - 1) * (total_steps // args.epochs))
                        n_steps = global_step
                        elapsed = time.time() - t0
                        eta_s = (total_steps - n_steps) * elapsed / max(n_steps, 1)
                        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        log(f"[E{epoch:03d} {n_steps/total_steps*100:5.1f}% S{n_steps:05d}] "
                            f"loss={loss.item()*args.ga_steps:.4f} avg_loss={epoch_loss/max(1,step+1):.4f} "
                            f"lr={opt.param_groups[0]['lr']:.1e} mem={mem:.1f}GB eta={eta_s/3600:.1f}h")

                    if global_step % args.auto_save_interval == 0:
                        save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step,
                                        best_loss, metrics_history, log_dir / "latest_step.pt", step_loss=loss.item())
                        log_f.flush()

                    if global_step % args.save_interval == 0:
                        save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step,
                                        best_loss, metrics_history,
                                        log_dir / f"v5_{args.phase}_{args.modality}_step{global_step}.pt")
                        log(f"Saved: step {global_step}")

            except Exception as e:
                log(f"ERROR at step {step}: {e}")
                import traceback; traceback.print_exc()
                save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step,
                                best_loss, metrics_history, log_dir / "latest_step.pt", step_loss=0)
                log_f.flush()
                raise

        if ga_count > 0:
            global_step += 1
            if scaler:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()
            opt.zero_grad()

        log(f"Epoch {epoch} done. avg_loss: {epoch_loss/max(1,step+1):.4f}")

    log("Training finished.")
    log_f.flush()
    log_f.close()


def main():
    p = argparse.ArgumentParser("V5 Training")
    p.add_argument("--phase", type=str, default="encoder",
                   choices=["encoder", "projector", "qwen"],
                   help="encoder: contrastive (no Qwen) | projector: align projection | qwen: LoRA LM")
    p.add_argument("--modality", type=str, default="all",
                   choices=["t1", "flair", "t2", "all"])
    p.add_argument("--data_root", type=str, default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--qwen_path", type=str, default="/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct")
    p.add_argument("--log_dir", type=str, default="outputs/report_gen")
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ga_steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_drop", type=float, default=0.05)
    p.add_argument("--max_text_len", type=int, default=512)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--auto_save_interval", type=int, default=100)
    p.add_argument("--eval_samples", type=int, default=100)
    p.add_argument("--batch_id", type=str, default=None, help="Filter to specific batch(s), comma-separated (e.g. batch00,batch01)")
    p.add_argument("--init_from", type=str, default=None, help="Path to shared initial weights for FedAvg (ignored if auto_resume finds checkpoint)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
