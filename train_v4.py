import argparse, os, sys, time, json, pickle, traceback, signal
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoder_v4 import ReportingModelV4, UniFormerEncoder
from mrrate_dataset import MRRateDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100


def alignment_loss(vt, text_embeds, tok, max_text_len=128):
    B = vt.shape[0]
    sparse = text_embeds if text_embeds.is_sparse else False
    vt_p = vt.float().mean(dim=1)
    te = text_embeds[:, :max_text_len].float()
    if te.shape[1] == 0:
        return torch.tensor(0.0, device=vt.device, requires_grad=True)
    mask = (te.sum(dim=-1) != 0).float().unsqueeze(-1)
    te_m = (te * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.mse_loss(vt_p, te_m)


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


def build_embeddings(llm, tok, vt_float, target_ids, B, n_vt, device):
    text_embeds = llm.get_input_embeddings()(target_ids)
    combined = torch.cat([vt_float.to(text_embeds.dtype), text_embeds], dim=1)
    ignore_labels = torch.full((B, n_vt), IGNORE_INDEX, dtype=target_ids.dtype, device=device)
    labels = torch.cat([ignore_labels, target_ids], dim=1)
    attn_mask = torch.cat([
        torch.ones(B, n_vt, device=device, dtype=torch.long),
        (target_ids != tok.pad_token_id).long()], dim=1)
    return combined, labels, attn_mask


def save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step,
                    best_loss, metrics, path, step_loss=None):
    state = {
        "encoder_state": enc.state_dict(),
        "llm_state": llm.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch, "global_step": global_step,
        "best_loss": best_loss, "metrics": metrics,
        "step_loss": step_loss,
    }
    torch.save(state, path)


def load_checkpoint(path, enc, llm, opt, sched, scaler, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    enc.load_state_dict(ckpt["encoder_state"], strict=False)
    llm.load_state_dict(ckpt["llm_state"], strict=False)
    try:
        opt.load_state_dict(ckpt["optimizer_state"])
    except ValueError:
        print(f"  [WARN] Skipping optimizer state (param group mismatch — different training phase)")
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
    log_path = log_dir / f"train_v4_{args.modality}_{ts}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    crash_reported = [False]
    def crash_handler(sig, frame):
        if crash_reported[0]:
            sys.exit(1)
        crash_reported[0] = True
        sig_name = signal.Signals(sig).name
        log(f"FATAL: received {sig_name} (signal {sig})")
        log(traceback.format_stack(frame))
        p = log_dir / f"latest_step.pt"
        save_checkpoint(enc, llm, opt, sched, scaler, -1, global_step, best_loss, metrics_history, p, step_loss=0)
        log(f"  Emergency checkpoint saved -> {p.name}")
        log_f.flush()
        sys.exit(1)

    def check_cuda():
        err = torch.cuda.synchronize(dev)
        if err is not None:
            log(f"CUDA ERROR: {err}")

    log(f"V4 Log: {log_path}")
    log(f"Backbone: BrainMVP UniFormer-Small | projector={args.projector} | brainmvp_ckpt={args.brainmvp_ckpt}")

    train_ds = MRRateDataset(args.data_root, "train", normalize="zscore", augment=args.augment)
    val_ds = MRRateDataset(args.data_root, "val", normalize="zscore", augment=False)
    if args.max_samples:
        train_ds.samples = train_ds.samples.iloc[:args.max_samples]
        val_ds.samples = val_ds.samples.iloc[:max(1, args.max_samples // 10)]

    dl_kw = dict(shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers,
                  pin_memory=False, persistent_workers=False, drop_last=True)
    if args.num_workers > 0:
        dl_kw["prefetch_factor"] = 2
        dl_kw["multiprocessing_context"] = "spawn"
    else:
        dl_kw["pin_memory"] = True
    tl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, **dl_kw)

    vl = None
    if args.eval_samples:
        vl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    tok = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    G = args.grid
    modality = args.modality
    is_single = modality != "all"
    n_vt_per_mod = 4 * (G ** 3)
    n_vt = n_vt_per_mod if is_single else 3 * n_vt_per_mod

    if is_single:
        enc = UniFormerEncoder(projector=args.projector, grid=G,
                                vit_dim=args.vit_dim, vit_heads=args.vit_heads,
                                vit_depth=args.vit_depth, out_dim=args.llm_dim)
        if args.brainmvp_ckpt:
            enc.load_brainmvp_weights(args.brainmvp_ckpt)
        enc.to(dev)
        log(f"Modality: {modality.upper()}  (single encoder)  n_vt={n_vt}")
    else:
        enc = ReportingModelV4(projector=args.projector, llm_dim=args.llm_dim, grid=G,
                                vit_dim=args.vit_dim, vit_heads=args.vit_heads,
                                vit_depth=args.vit_depth, use_compile=args.compile,
                                brainmvp_ckpt=args.brainmvp_ckpt).to(dev)
        log(f"Modality: ALL  (3 encoders)  n_vt={n_vt}")

    uni_params, other_params = [], []
    for name, p in enc.named_parameters():
        if "uniformer" in name:
            uni_params.append(p)
        else:
            other_params.append(p)

    log(f"UniFormer params (lr={args.cnn_lr}): {sum(p.numel() for p in uni_params):,}")
    log(f"Projector params  (lr={args.lr}): {sum(p.numel() for p in other_params):,}")

    if args.phase == "uniformer":
        for p in other_params:
            p.requires_grad = False
        log(f"Phase: uniformer — frozen projector ({sum(p.numel() for p in other_params):,} params)")
    elif args.phase == "projector":
        for p in uni_params:
            p.requires_grad = False
        log(f"Phase: projector — frozen UniFormer ({sum(p.numel() for p in uni_params):,} params)")

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen_path, torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True)
    llm.resize_token_embeddings(len(tok))

    if args.train_mode == "align":
        for p in llm.parameters():
            p.requires_grad = False
        llm_log = f"Qwen: embedding only (no LoRA, no forward — align mode)"
        log(llm_log)
    else:
        llm.gradient_checkpointing_enable()
        llm.config.use_cache = False
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_drop,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                              bias="none", task_type="CAUSAL_LM")
        llm = get_peft_model(llm, lora_cfg).to(dev)
        llm_log = f"Qwen LoRA params: {sum(p.numel() for p in llm.parameters() if p.requires_grad):,}"

    if args.phase in ("uniformer", "projector"):
        for p in llm.parameters():
            p.requires_grad = False
        llm_log += " [FROZEN]"

    log(llm_log)

    trainable_params = []
    for pg in [uni_params, other_params]:
        trainable_g = [p for p in pg if p.requires_grad]
        if trainable_g:
            lr = args.cnn_lr if pg is uni_params else args.lr
            trainable_params.append({"params": trainable_g, "lr": lr})

    llm_trainable = [p for p in llm.parameters() if p.requires_grad]
    if llm_trainable:
        trainable_params.append({"params": llm_trainable, "lr": args.lr})

    total_trainable = sum(len(g["params"]) for g in trainable_params)
    log(f"Total trainable param groups: {len(trainable_params)}")

    opt = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay, fused=True)

    steps_per_epoch = len(tl) // args.ga_steps
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    signal.signal(signal.SIGTERM, crash_handler)
    signal.signal(signal.SIGSEGV, crash_handler)

    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}  Workers: {args.num_workers}")
    log(f"Tokens: {n_vt}  Epochs: {args.epochs}  St/epoch: {steps_per_epoch}  Total: {total_steps}")
    log(f"Phase: {args.phase}  LR(encoder)={args.lr} LR(uniformer)={args.cnn_lr}  AMP: {args.use_amp}  GA: {args.ga_steps}")
    log("=" * 55)

    start_epoch, global_step, best_loss = 1, 0, float("inf")
    metrics_history = []

    if args.resume and Path(args.resume).exists():
        log(f"Resuming from: {args.resume}")
        start_epoch, global_step, best_loss, metrics_history = load_checkpoint(
            args.resume, enc, llm, opt, sched, scaler, dev)
        log(f"  epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")
    elif args.auto_resume:
        auto_ckpt = log_dir / "latest_step.pt"
        if auto_ckpt.exists():
            log(f"Auto-resuming from: {auto_ckpt}")
            start_epoch, global_step, best_loss, metrics_history = load_checkpoint(
                auto_ckpt, enc, llm, opt, sched, scaler, dev)
            log(f"  epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        enc.train(); llm.train()
        epoch_loss, epoch_steps = 0.0, 0
        t0 = time.time()
        opt.zero_grad()

        for step, batch in enumerate(tl):
            if args.resume and epoch == start_epoch and step < global_step % len(tl):
                continue
            try:
                t1 = batch["t1"].to(dev, non_blocking=True)
                flair = batch["flair"].to(dev, non_blocking=True)
                t2 = batch["t2"].to(dev, non_blocking=True)
                h1, hf, h2 = batch["has_t1"], batch["has_flair"], batch["has_t2"]
                B = t1.shape[0]

                if is_single:
                    mod_map = {"t1": t1, "flair": flair, "t2": t2}
                    vol = mod_map[modality]
                    vt = enc(vol)
                else:
                    vt = enc(t1, flair, t2, h1, hf, h2)
                target_ids = tok(batch["reports"], return_tensors="pt",
                                 padding=True, truncation=True, max_length=1024)["input_ids"].to(dev)

                if args.train_mode == "align":
                    text_embeds = llm.get_input_embeddings()(target_ids)
                    loss = alignment_loss(vt, text_embeds, tok) / args.ga_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    embeds, labels, am = build_embeddings(llm, tok, vt, target_ids, B, n_vt, dev)
                    if scaler:
                        with torch.amp.autocast("cuda"):
                            loss = llm(inputs_embeds=embeds, attention_mask=am, labels=labels).loss / args.ga_steps
                        scaler.scale(loss).backward()
                    else:
                        loss = llm(inputs_embeds=embeds, attention_mask=am, labels=labels).loss / args.ga_steps
                        loss.backward()

                step_loss = loss.item() * args.ga_steps
                epoch_loss += step_loss; epoch_steps += 1
            except Exception as ex:
                log(f"ERROR at step {step}: {type(ex).__name__}: {ex}")
                log(traceback.format_exc())
                crash_path = log_dir / f"crash_epoch{epoch:03d}_step{global_step:06d}.pt"
                save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, crash_path, step_loss=0)
                log(f"  Crash checkpoint saved -> {crash_path.name}")
                p = log_dir / "latest_step.pt"
                save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, p, step_loss=0)
                log_f.flush()
                raise

            should_step = (step + 1) % args.ga_steps == 0
            if should_step:
                global_step += 1
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], args.max_grad_norm)
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], args.max_grad_norm)
                    opt.step()
                opt.zero_grad()
                sched.step()

            if should_step and global_step % args.log_interval == 0:
                lrs = [g["lr"] for g in opt.param_groups]
                pct = 100.0 * step / len(tl)
                elapsed = time.time() - t0
                est_total = elapsed / max(step, 1) * len(tl)
                eta_s = est_total - elapsed
                eta_str = f"{eta_s:.0f}s" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"
                log(f"[E{epoch:03d} {pct:5.1f}% S{global_step:05d}] "
                    f"loss={step_loss:.4f} lr={lrs[0]:.2e}/{lrs[1]:.2e} "
                    f"mem={torch.cuda.memory_allocated(dev) / 1e9:.1f}GB eta={eta_str}")

            if should_step and global_step % args.save_interval == 0:
                p = log_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, p, step_loss=step_loss)
                log(f"  Saved {p.name}")

            if should_step and global_step % args.auto_save_interval == 0:
                p = log_dir / "latest_step.pt"
                save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, p, step_loss=step_loss)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        log(f"--- Epoch {epoch:03d} loss={avg_loss:.4f} time={elapsed / 3600:.1f}h ---")

        epoch_metrics = {"epoch": epoch, "loss": avg_loss, "time_h": elapsed / 3600, "global_step": global_step}

        if vl and args.eval_samples:
            try:
                from eval_report import run_eval
                log("[Evaluating] ...")
                eval_t0 = time.time()
                eval_metrics, preds, refs = run_eval(enc, llm, tok, vl, dev, n_vt,
                    max_samples=args.eval_samples, max_new_tokens=256)
                eval_metrics["eval_time_s"] = time.time() - eval_t0
                epoch_metrics["eval"] = eval_metrics
                log(f"  BLEU-1={eval_metrics.get('bleu1',0):.4f} BLEU-4={eval_metrics.get('bleu4',0):.4f} "
                    f"ROUGE-L={eval_metrics.get('rougeL',0):.4f} METEOR={eval_metrics.get('meteor',0):.4f}")
                if preds and refs:
                    preds_file = log_dir / f"eval_v4_epoch{epoch:03d}_preds.json"
                    json.dump({"predictions": preds, "references": refs}, preds_file.open("w"), indent=2, ensure_ascii=False)
                    log(f"  Preds saved -> {preds_file.name}")
            except ImportError:
                log("[Evaluating] SKIPPED")

        metrics_history.append(epoch_metrics)
        last_path = log_dir / "last_model.pt"
        save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, last_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = log_dir / "best_model_v4.pt"
            save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, best_path)
            log(f"  >> New best: {best_loss:.4f}")

        epoch_ckpt = log_dir / f"epoch_v4_{epoch:03d}.pt"
        save_checkpoint(enc, llm, opt, sched, scaler, epoch, global_step, best_loss, metrics_history, epoch_ckpt)

    log(f"Done! Best loss: {best_loss:.4f}")
    metrics_path = log_dir / "metrics_v4.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    log(f"Metrics saved to {metrics_path}")
    log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--brainmvp_ckpt", type=str, default="",
                   help="BrainMVP pretrained checkpoint path")
    p.add_argument("--train_mode", type=str, default="qwen", choices=["qwen", "align"],
                   help="qwen: full LM loss (slow), align: MSE between vision tokens and text embeddings (fast)")
    p.add_argument("--projector", type=str, default="attn", choices=["attn", "mlp"],
                   help="Projector type: attn (Self-Attention fusion) or mlp (Qwen3-VL style 2-layer MLP)")
    p.add_argument("--modality", type=str, default="all", choices=["t1", "flair", "t2", "all"],
                   help="Train single modality encoder only (t1/flair/t2) or all three (all)")
    p.add_argument("--phase", type=str, default="all", choices=["uniformer", "projector", "all"],
                   help="Training phase: uniformer (freeze projector+LLM), projector (freeze UniFormer+LLM), all")
    p.add_argument("--augment", action="store_true", default=False,
                   help="Enable online data augmentation (random flip + intensity noise)")
    p.add_argument("--qwen_path", default="/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct")
    p.add_argument("--log_dir", default="outputs/report_gen")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cnn_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ga_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=200,
                   help="Save full checkpoint every N steps (named step_N.pt)")
    p.add_argument("--auto_save_interval", type=int, default=50,
                   help="Save rolling recovery checkpoint every N steps (latest_step.pt)")
    p.add_argument("--auto_resume", action="store_true", default=False,
                   help="Auto-resume from latest_step.pt if it exists")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--eval_samples", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_drop", type=float, default=0.1)
    p.add_argument("--llm_dim", type=int, default=2048)
    p.add_argument("--grid", type=int, default=2)
    p.add_argument("--vit_dim", type=int, default=512)
    p.add_argument("--vit_heads", type=int, default=8)
    p.add_argument("--vit_depth", type=int, default=2)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
