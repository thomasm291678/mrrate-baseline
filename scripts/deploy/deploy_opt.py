import paramiko, io, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

LOCAL = r"C:\Users\HP\Documents\5555"
FILES = [
    ("encoder_v5.py", f"{LOCAL}/encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py"),
    ("mrrate_dataset", f"{LOCAL}/server_code/mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py"),
    ("phase1", f"{LOCAL}/train_v5_phase1.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase1.py"),
    ("phase2", f"{LOCAL}/train_v5_phase2.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase2.py"),
    ("phase3", f"{LOCAL}/train_v5_phase3.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase3.py"),
]

# Upload
sftp = c.open_sftp()
for name, local, remote in FILES:
    with open(local, "rb") as f:
        sftp.putfo(io.BytesIO(f.read()), remote)
    print(f"[1] {name} uploaded")
sftp.close()

# Kill old phase3 + tmux
c.exec_command("pkill -9 -f train_v5_phase3 2>/dev/null || true")
c.exec_command("tmux kill-session -t phase3 2>/dev/null || true")
time.sleep(2)

# Write launch script
launch = r"""#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
CUDA_VISIBLE_DEVICES=0 /home/jiaqigu/hidnet_env/bin/python -u scripts/train_v5_phase3.py \
  --encoder_ckpt outputs/report_gen/phase2_latest.pt \
  --batch_id batch27 --epochs 3 \
  --batch_size 8 --num_workers 8 --prefetch_factor 4 \
  --lr 5e-5 --wd 0.01 \
  --lora_r 8 --lora_alpha 16 \
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --log_dir outputs/report_gen \
  --log_interval 5 --auto_save_interval 50 \
  --auto_resume --use_amp
"""

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(launch.encode()), "/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3_opt.sh")
sftp.close()
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/scripts/launch_phase3_opt.sh")
print("[2] Launch script written")

# Launch in tmux
s, o, e = c.exec_command("tmux new-session -d -s phase3 '/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3_opt.sh 2>&1 | tee /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_opt.log'", timeout=10)
time.sleep(3)
s, o, e = c.exec_command("tmux has-session -t phase3 && echo SESSION_OK || echo NO_SESSION")
print("[3] Tmux:", o.read().decode().strip())

print("[4] Waiting 60s for Qwen to load with torch.compile...")
time.sleep(60)

# Check logs
s, o, e = c.exec_command("ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_qwen_*.log 2>/dev/null | head -1")
logf = o.read().decode().strip()
if logf:
    s2, o2, e2 = c.exec_command(f"tail -10 {logf}")
    print("\n--- Latest phase3 log ---")
    print(o2.read().decode())

# GPU
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1")
print(f"\nGPU0: {o.read().decode().strip()}")

c.close()
print("\nOptimized Phase 3 deployed! Check: tmux attach -t phase3")
