"""
Train: frozen DenseNet encoders + trainable projections + Qwen LoRA → report generation.

Stage: align projections to Qwen embedding space using real MR-RATE reports.
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mrrate_dataset import MRRateDataset
from scripts.merge_lightweight import LightweightMultiModal
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"


def collate_fn(batch):
    """Pad and collate MR-RATE samples for report generation."""
    t1 = torch.stack([b["t1"] for b in batch])
    flair = torch.stack([b["flair"] for b in batch])
    t2 = torch.stack([b["t2"] for b in batch])
    h1 = torch.stack([b["has_t1"] for b in batch])
    hf = torch.stack([b["has_flair"] for b in batch])
    h2 = torch.stack([b["has_t2"] for b in batch])
    reports = [b["report"] for b in batch]
    return {"t1": t1, "flair": flair, "t2": t2,
            "has_t1": h1, "has_flair": hf, "has_t2": h2,
            "reports": reports}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data_root = args.data_root
    train_ds = MRRateDataset(data_root, "train")
    if args.max_samples > 0:
        train_ds.samples = train_ds.samples.iloc[:args.max_samples]
    val_ds = MRRateDataset(data_root, "val")
    if args.max_samples > 0:
        val_ds.samples = val_ds.samples.iloc[:max(1, args.max_samples // 10)]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # Tokenizer
    qwen_path = args.qwen_path
    tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    img_tok_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)

    # Encoder (frozen)
    print("Loading encoder...")
    encoder = LightweightMultiModal(freeze_encoders=True).to(device)
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt)
    encoder.eval()
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params "
          f"({sum(p.numel() for p in encoder.parameters() if p.requires_grad):,} trainable)")

    # Qwen (LoRA)
    print("Loading Qwen...")
    llm = AutoModelForCausalLM.from_pretrained(
        qwen_path, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
    llm.resize_token_embeddings(len(tok))
    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable()
        llm.config.use_cache = False
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM")
    llm = get_peft_model(llm, lora_cfg).to(device)
    print(f"  Qwen LoRA trainable: {sum(p.numel() for p in llm.parameters() if p.requires_grad):,}")

    # Resume
    start_epoch = 1
    start_step = 0
    if args.resume:
        ckpt2 = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt2["encoder_state"])
        llm.load_state_dict(ckpt2["llm_state"], strict=False)
        start_epoch = ckpt2.get("epoch", 0) + 1
        start_step = ckpt2.get("step", 0)
        # optimizer/scheduler state loaded after creation
        global_step = start_step
        print(f"Resumed from epoch {start_epoch}, step {start_step}")

    # Optimizer: only trainable params (projections + LoRA)
    trainable = [p for n, p in list(encoder.named_parameters()) + list(llm.named_parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1),
                                                num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    # Resume optimizer and scheduler state
    if args.resume:
        ckpt2 = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer_state" in ckpt2:
            optimizer.load_state_dict(ckpt2["optimizer_state"])
        if "scheduler_state" in ckpt2:
            scheduler.load_state_dict(ckpt2["scheduler_state"])

    log_dir = Path("outputs/report_gen")
    log_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"  Report Generation Training")
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Batch: {args.batch_size}")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  AMP: {args.use_amp}")
    print(f"{'='*60}\n")

    global_step = start_step
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        llm.train()
        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            t1 = batch["t1"].to(device)
            flair = batch["flair"].to(device)
            t2 = batch["t2"].to(device)
            h1 = batch["has_t1"]
            hf = batch["has_flair"]
            h2 = batch["has_t2"]
            B = t1.shape[0]

            # Encode images
            with torch.no_grad():
                visual_tokens = encoder(t1, flair, t2, h1, hf, h2)  # (B, 12, 2048)
            n_vt = visual_tokens.shape[1]

            # Tokenize reports
            reports = batch["reports"]
            target_texts = [r[:1024] for r in reports]
            target_ids = tok(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)["input_ids"].to(device)

            # Prepend visual tokens as embeddings: [visual_tokens | text_embeddings]
            text_embeds = llm.get_input_embeddings()(target_ids)
            vt_float = visual_tokens.to(text_embeds.dtype)
            # Create labels: visual token positions → IGNORE
            combined_embeds = torch.cat([vt_float, text_embeds], dim=1)
            ignore_labels = torch.full((B, n_vt), IGNORE_INDEX, dtype=target_ids.dtype, device=device)
            labels = torch.cat([ignore_labels, target_ids], dim=1)
            attn_mask = torch.cat([
                torch.ones(B, n_vt, device=device, dtype=torch.long),
                (target_ids != tok.pad_token_id).long(),
            ], dim=1)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = llm(inputs_embeds=combined_embeds, attention_mask=attn_mask, labels=labels)
                    loss = out.loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out = llm(inputs_embeds=combined_embeds, attention_mask=attn_mask, labels=labels)
                loss = out.loss / args.grad_accum_steps
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * args.grad_accum_steps
            global_step += 1

            if step % args.log_interval == 0:
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  [E{epoch:03d} S{step:04d}] loss={loss.item() * args.grad_accum_steps:.4f} lr={scheduler.get_last_lr()[0]:.2e} mem={mem:.1f}GB")


            if step > 0 and step % 500 == 0:
                torch.save({"encoder_state": encoder.state_dict(),
                            "llm_state": llm.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "epoch": epoch, "step": global_step,
                            "loss": loss.item() * args.grad_accum_steps,
                            "best_loss": best_loss}, log_dir / "best_model.pt")
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        # Validation
        val_loss = 0.0
        if len(val_loader) > 0:
            encoder.eval()
            llm.eval()
            with torch.no_grad():
                for vbatch in val_loader:
                    vt = torch.cat([vbatch["t1"].to(device), vbatch["flair"].to(device), vbatch["t2"].to(device)])
                    # Simplified val: just compute reconstruction loss on a prompt
                    prompts = [f"{IMAGE_TOKEN}\n"] * vt.shape[0]
                    pids = tok(prompts, return_tensors="pt", padding=True)["input_ids"].to(device)
                    # ... (simplified for now)
                    break  # Full val would need proper batching
            val_loss /= max(len(val_loader), 1)

        print(f"--- Epoch {epoch:03d} train_loss={avg_loss:.4f} time={elapsed:.0f}s ---")

        # Save every epoch for safety
        torch.save({"encoder_state": encoder.state_dict(),
                    "llm_state": llm.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "loss": avg_loss}, log_dir / "last_model.pt")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"encoder_state": encoder.state_dict(),
                        "llm_state": llm.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "loss": avg_loss,
                        "best_loss": best_loss}, log_dir / "best_model.pt")
            print(f"  >> Saved best (loss={best_loss:.4f}, step={global_step})")

    print(f"\nDone! Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/nas1/disk07/public/mr_data/MR-RATE")
    parser.add_argument("--encoder_ckpt", default="outputs/pretrain_densenet/multimodal_encoder.pt")
    parser.add_argument("--qwen_path", default="/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max training samples (0=all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
