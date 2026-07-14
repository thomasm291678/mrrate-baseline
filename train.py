import argparse, os, sys, time, json, pickle
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder import ReportingModel
from mrrate_dataset import MRRateDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100


def collate_fn(batch):
    t1_list, flair_list, t2_list = [], [], []
    h1_list, hf_list, h2_list = [], [], []
    reports_list, tok_list = [], []

    for b in batch:
        t1_list.append(b["t1"])
        flair_list.append(b["flair"])
        t2_list.append(b["t2"])
        h1_list.append(b["has_t1"])
        hf_list.append(b["has_flair"])
        h2_list.append(b["has_t2"])
        reports_list.append(b["report"])
        tok_list.append(b.get("tokens"))

    result = {
        "t1": torch.stack(t1_list),
        "flair": torch.stack(flair_list),
        "t2": torch.stack(t2_list),
        "has_t1": torch.stack(h1_list),
        "has_flair": torch.stack(hf_list),
        "has_t2": torch.stack(h2_list),
        "reports": reports_list,
    }

    if all(t is not None for t in tok_list):
        max_len = max(t.shape[0] for t in tok_list)
        padded = torch.zeros(len(tok_list), max_len, dtype=torch.long)
        for i, t in enumerate(tok_list):
            padded[i, :t.shape[0]] = t
        result["token_ids"] = padded
        result["tokenized"] = True
    else:
        result["tokenized"] = False

    return result


def build_embeddings(llm, tok, vt_float, target_ids, B, n_vt, device):
    text_embeds = llm.get_input_embeddings()(target_ids)
    combined = torch.cat([vt_float.to(text_embeds.dtype), text_embeds], dim=1)
    ignore_labels = torch.full((B, n_vt), IGNORE_INDEX, dtype=target_ids.dtype,
                               device=device)
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
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "metrics": metrics,
        "step_loss": step_loss,
    }
    torch.save(state, path)


def load_checkpoint(path, enc, llm, opt, sched, scaler, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    enc.load_state_dict(ckpt["encoder_state"], strict=False)
    llm.load_state_dict(ckpt["llm_state"], strict=False)
    opt.load_state_dict(ckpt["optimizer_state"])
    sched.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return (
        ckpt["epoch"],
        ckpt["global_step"],
        ckpt["best_loss"],
        ckpt.get("metrics", []),
    )


def pre_tokenize_dataset(dataset, tok, cache_path, log_fn):
    if Path(cache_path).exists():
        log_fn(f"Loading tokenized cache from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log_fn(f"Pre-tokenizing {len(dataset)} reports (one-time)...")
    tokenized = {}
    t0 = time.time()
    for i in range(len(dataset)):
        row = dataset.samples.iloc[i]
        uid = str(row["study_uid"])
        report = dataset.reports.get(uid, "")
        encoded = tok(report, return_tensors="pt", truncation=True,
                      max_length=1024)["input_ids"].squeeze(0)
        tokenized[uid] = encoded

        if (i + 1) % 5000 == 0:
            pct = 100.0 * (i + 1) / len(dataset)
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            log_fn(f"  tokenized {i+1}/{len(dataset)} ({pct:.0f}%) {speed:.0f}/s")

    elapsed = time.time() - t0
    log_fn(f"Pre-tokenization done in {elapsed:.0f}s ({len(tokenized)} entries)")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(tokenized, f)
    log_fn(f"Cache saved to {cache_path}")
    return tokenized


def train(args):
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        t = time.strftime("%H:%M:%S")
        line = f"[{t}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log(f"Log: {log_path}")
    log(f"torch.compile: {args.compile}  batch_size: {args.batch_size}  workers: {args.num_workers}")

    train_ds = MRRateDataset(args.data_root, "train")
    val_ds = MRRateDataset(args.data_root, "val")
    if args.train_frac:
        n_train = int(len(train_ds) * args.train_frac)
        train_ds.samples = train_ds.samples.iloc[:n_train]
        log(f"Using {args.train_frac*100:.0f}% training data: {n_train} samples")

    if args.max_samples:
        train_ds.samples = train_ds.samples.iloc[:args.max_samples]
        val_ds.samples = val_ds.samples.iloc[:max(1, args.max_samples // 10)]

    dl_kwargs = dict(
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True)
    tl = torch.utils.data.DataLoader(train_ds, **dl_kwargs)

    vl = None
    if args.eval_samples:
        val_kwargs = dict(
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
            pin_memory=True, drop_last=False)
        vl = torch.utils.data.DataLoader(val_ds, **val_kwargs)

    tok = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    token_cache = None
    if args.pre_tokenize:
        cache_path = log_dir / "token_cache.pkl"
        token_cache = pre_tokenize_dataset(train_ds, tok, cache_path, log)
        train_ds.token_cache = token_cache

    G = args.grid
    n_vt = 3 * 4 * (G ** 3)
    enc = ReportingModel(llm_dim=args.llm_dim, grid=G,
                         vit_dim=args.vit_dim, vit_heads=args.vit_heads,
                         vit_depth=args.vit_depth, freeze_cnn=False,
                         use_compile=args.compile).to(dev)

    cnn_params, other_params = [], []
    for name, p in enc.named_parameters():
        if not p.requires_grad:
            continue
        if "densenet" in name:
            cnn_params.append(p)
        else:
            other_params.append(p)

    log(f"DenseNet params (lr={args.cnn_lr}): {sum(p.numel() for p in cnn_params):,}")
    log(f"Other params     (lr={args.lr}): {sum(p.numel() for p in other_params):,}")

    if args.v1_ckpt and Path(args.v1_ckpt).exists():
        ckpt = torch.load(args.v1_ckpt, map_location=dev, weights_only=False)
        enc.load_densenet_weights(ckpt.get("encoder_state", ckpt))
        log("Loaded DenseNet backbone from v1 checkpoint")
    else:
        log("WARNING: no v1 checkpoint")

    llm = AutoModelForCausalLM.from_pretrained(
        args.qwen_path, torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True)
    llm.resize_token_embeddings(len(tok))
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM")
    llm = get_peft_model(llm, lora_cfg).to(dev)
    log(f"Qwen LoRA trainable: {sum(p.numel() for p in llm.parameters() if p.requires_grad):,}")

    if args.v1_ckpt and Path(args.v1_ckpt).exists():
        ckpt = torch.load(args.v1_ckpt, map_location=dev, weights_only=False)
        if "llm_state" in ckpt:
            llm_state = ckpt["llm_state"]
            emb_key = "base_model.model.model.embed_tokens.weight"
            lmh_key = "base_model.model.lm_head.weight"
            tok_len = len(tok)
            for key in [emb_key, lmh_key]:
                if key in llm_state and llm_state[key].shape[0] != tok_len:
                    ckpt_vocab = llm_state[key].shape[0]
                    log(f"Resizing {key}: ckpt={ckpt_vocab} -> tokenizer={tok_len}")
                    if ckpt_vocab > tok_len:
                        llm_state[key] = llm_state[key][:tok_len, :]
                    else:
                        pad = torch.zeros(tok_len - ckpt_vocab, llm_state[key].shape[1],
                                          dtype=llm_state[key].dtype, device=dev)
                        llm_state[key] = torch.cat([llm_state[key], pad], dim=0)
            llm.load_state_dict(llm_state, strict=False)
            log("Loaded Qwen LoRA from v1")

    opt = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": cnn_params, "lr": args.cnn_lr},
        {"params": [p for p in llm.parameters() if p.requires_grad],
         "lr": args.lr},
    ], weight_decay=args.weight_decay, fused=True)

    steps_per_epoch = len(tl) // args.ga_steps
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}  Workers: {args.num_workers}")
    log(f"Tokens: {n_vt}  Epochs: {args.epochs}  St/epoch: {steps_per_epoch}  Total: {total_steps}")
    log(f"LR(encoder)={args.lr} LR(cnn)={args.cnn_lr}  AMP: {args.use_amp}  GA: {args.ga_steps}")
    log("=" * 55)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    metrics_history = []

    if args.resume and Path(args.resume).exists():
        log(f"Resuming from checkpoint: {args.resume}")
        start_epoch, global_step, best_loss, metrics_history = load_checkpoint(
            args.resume, enc, llm, opt, sched, scaler, dev)
        log(f"  Resume at epoch={start_epoch} step={global_step} best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        enc.train()
        llm.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()
        opt.zero_grad()

        for step, batch in enumerate(tl):
            if args.resume and epoch == start_epoch and step < global_step % len(tl):
                continue

            t1 = batch["t1"].to(dev, non_blocking=True)
            flair = batch["flair"].to(dev, non_blocking=True)
            t2 = batch["t2"].to(dev, non_blocking=True)
            h1, hf, h2 = batch["has_t1"], batch["has_flair"], batch["has_t2"]
            B = t1.shape[0]

            vt = enc(t1, flair, t2, h1, hf, h2)

            if token_cache:
                target_ids = batch["token_ids"].to(dev, non_blocking=True)
            else:
                target_ids = tok(batch["reports"], return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=1024)["input_ids"].to(dev)

            embeds, labels, am = build_embeddings(
                llm, tok, vt, target_ids, B, n_vt, dev)

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = llm(inputs_embeds=embeds, attention_mask=am,
                               labels=labels).loss / args.ga_steps
                scaler.scale(loss).backward()
            else:
                loss = llm(inputs_embeds=embeds, attention_mask=am,
                           labels=labels).loss / args.ga_steps
                loss.backward()

            step_loss = loss.item() * args.ga_steps
            epoch_loss += step_loss
            epoch_steps += 1

            should_step = (step + 1) % args.ga_steps == 0
            if should_step:
                global_step += 1
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"],
                                                   args.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"],
                                                   args.max_grad_norm)
                    opt.step()
                opt.zero_grad()
                sched.step()

            if should_step and global_step % args.log_interval == 0:
                lrs = [g["lr"] for g in opt.param_groups]
                pct = 100.0 * step / len(tl)
                elapsed = time.time() - t0
                est_total = elapsed / max(step, 1) * len(tl)
                eta_s = est_total - elapsed
                if eta_s < 3600:
                    eta_str = f"{eta_s:.0f}s"
                else:
                    eta_str = f"{eta_s/3600:.1f}h"
                log(f"[E{epoch:03d} {pct:5.1f}% S{global_step:05d}] "
                    f"loss={step_loss:.4f} "
                    f"lr={lrs[0]:.2e}/{lrs[1]:.2e} "
                    f"mem={torch.cuda.memory_allocated(dev)/1e9:.1f}GB "
                    f"eta={eta_str}")

            if should_step and global_step % args.save_interval == 0:
                ckpt_path = log_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(enc, llm, opt, sched, scaler, epoch,
                                global_step, best_loss, metrics_history,
                                ckpt_path, step_loss=step_loss)
                log(f"  Saved {ckpt_path.name}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        log(f"--- Epoch {epoch:03d} loss={avg_loss:.4f} time={elapsed/3600:.1f}h ---")

        epoch_metrics = {
            "epoch": epoch, "loss": avg_loss,
            "time_h": elapsed / 3600, "global_step": global_step,
        }

        if vl and args.eval_samples:
            try:
                from eval_report import run_eval  # noqa: F811
                log("[Evaluating] Generating validation reports...")
                eval_t0 = time.time()
                eval_metrics, preds, refs = run_eval(
                    enc, llm, tok, vl, dev, n_vt,
                    max_samples=args.eval_samples, max_new_tokens=256)
                eval_metrics["eval_time_s"] = time.time() - eval_t0
                epoch_metrics["eval"] = eval_metrics
                log(
                    f"  BLEU-1={eval_metrics.get('bleu1',0):.4f} "
                    f"BLEU-4={eval_metrics.get('bleu4',0):.4f} "
                    f"ROUGE-L={eval_metrics.get('rougeL',0):.4f} "
                    f"METEOR={eval_metrics.get('meteor',0):.4f} "
                    f"BERT-F1={eval_metrics.get('bert_f1',0):.4f}")
                log(
                    f"  Diversity: unique={eval_metrics.get('unique_count',0)}/{eval_metrics.get('total',0)} "
                    f"uniqueness={eval_metrics.get('uniqueness',0):.3f} "
                    f"pair_sim={eval_metrics.get('pairwise_sim',0):.3f}")
            except ImportError:
                log("[Evaluating] SKIPPED: eval_report dependencies missing")

            # Save predictions for unified evaluation
            if preds and refs:
                preds_file = log_dir / f"eval_epoch{epoch:03d}_preds.json"
                json.dump({"predictions": preds, "references": refs},
                          preds_file.open("w"), indent=2, ensure_ascii=False)
                log(f"  Predictions saved → {preds_file.name}")

        metrics_history.append(epoch_metrics)

        last_path = log_dir / "last_model.pt"
        save_checkpoint(enc, llm, opt, sched, scaler, epoch,
                        global_step, best_loss, metrics_history,
                        last_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = log_dir / "best_model.pt"
            save_checkpoint(enc, llm, opt, sched, scaler, epoch,
                            global_step, best_loss, metrics_history,
                            best_path)
            log(f"  >> New best: {best_loss:.4f}")

        epoch_ckpt = log_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(enc, llm, opt, sched, scaler, epoch,
                        global_step, best_loss, metrics_history,
                        epoch_ckpt)

    log(f"Done! Best loss: {best_loss:.4f}")
    metrics_path = log_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    log(f"Metrics saved to {metrics_path}")
    log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    p.add_argument("--v1_ckpt", default="outputs/report_gen/best_model.pt")
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
    p.add_argument("--pre_tokenize", action="store_true", default=False)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--eval_samples", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=1.0)
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
