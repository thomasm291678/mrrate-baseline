import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)
D = "/home/jiaqigu/mrrate_hidnet"

# Update loss function in train_v5_phase2.py
fix = """
def alignment_loss_fn(vt, text_embeds, enc, freq_weights):
    te = text_embeds.float()
    if te.shape[1] == 0:
        return torch.tensor(0.0, device=vt.device, requires_grad=True)
    te_37 = enc.disease_proj(te)
    w = freq_weights.to(vt.device).float().view(1, -1, 1)
    se = (vt.float() - te_37).pow(2)
    return (se * w).sum() / se.numel()
"""

# Read current file
_, o, _ = c.exec_command(f"cat {D}/train_v5_phase2.py")
content = o.read().decode()

# Replace the old loss fn
old_fn = """def alignment_loss_fn(vt, text_embeds, enc, freq_weights):
    te = text_embeds.float()
    if te.shape[1] == 0:
        return torch.tensor(0.0, device=vt.device, requires_grad=True)
    te_37 = enc.disease_proj(te)
    w = freq_weights.to(vt.device).float().view(1, 37, 1)
    return F.mse_loss(vt.float() * w, te_37 * w)"""

content = content.replace(old_fn, fix.strip())

sf = c.open_sftp()
import io
sf.putfo(io.BytesIO(content.encode()), f"{D}/train_v5_phase2.py")
sf.close()
print("Fixed loss function on farm05")

# Restart Phase 2
c.exec_command("pkill -9 -f train_v5_phase2 2>/dev/null; sleep 1")
c.exec_command(f"rm -f {D}/outputs/report_gen/phase2*.pt {D}/outputs/report_gen/phase2.log")

scr = f"""#!/bin/bash
source ~/hidnet_env/bin/activate
cd {D}
python train_v5_phase2.py \\
  --encoder_ckpt {D}/outputs/report_gen/phase1_latest.pt \\
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
  --log_dir {D}/outputs/report_gen \\
  --batch_id batch27 \\
  --epochs 3 \\
  --batch_size 4 \\
  --lr 1e-4 \\
  --max_text_len 256 \\
  2>&1 | tee outputs/report_gen/phase2.log
"""

sf = c.open_sftp()
sf.putfo(io.BytesIO(scr.encode()), f"{D}/run_phase2.sh")
sf.close()
c.exec_command(f"chmod +x {D}/run_phase2.sh")
_, o, _ = c.exec_command(f"cd {D} && nohup bash run_phase2.sh > /dev/null 2>&1 & echo PID=$!")
print("Restarted:", o.read().decode().strip())

c.close()
