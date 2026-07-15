import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Upload phase3 script
sftp = c.open_sftp()
sftp.put("train_v5_phase3.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase3.py")
sftp.put("encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
sftp.put("server_code/mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")
sftp.close()
print("[1] Code uploaded")

# Kill leftover
c.exec_command("pkill -9 -f train_v5_phase3 2>/dev/null || true")
time.sleep(1)

# Write launch script
launch_sh = r"""#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
CUDA_VISIBLE_DEVICES=0 /home/jiaqigu/hidnet_env/bin/python -u scripts/train_v5_phase3.py \
  --encoder_ckpt outputs/report_gen/phase2_latest.pt \
  --batch_id batch27 --epochs 3 --batch_size 2 --num_workers 2 \
  --lr 5e-5 --wd 0.01 \
  --lora_r 16 --lora_alpha 32 \
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --log_dir outputs/report_gen \
  --log_interval 10 --auto_save_interval 50 \
  --auto_resume --use_amp
"""

sftp = c.open_sftp()
sftp.putfo(
    __import__("io").BytesIO(launch_sh.encode()),
    "/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3.sh"
)
sftp.close()
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/scripts/launch_phase3.sh")
print("[2] Launch script written")

# Launch in tmux
s, o, e = c.exec_command("tmux kill-session -t phase3 2>/dev/null || true", timeout=5)
s, o, e = c.exec_command("tmux new-session -d -s phase3 '/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3.sh 2>&1 | tee /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_tmux.log'", timeout=10)
time.sleep(3)

# Verify tmux
s, o, e = c.exec_command("tmux has-session -t phase3 && echo SESSION_OK || echo NO_SESSION", timeout=5)
print("[3] Tmux:", o.read().decode().strip())

# Wait for model loading
print("[4] Waiting for Qwen to load (30s)...")
time.sleep(30)

# Check log
s, o, e = c.exec_command("tail -15 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_tmux.log 2>/dev/null || echo NO_LOG")
log_content = o.read().decode().strip()
if not log_content or log_content == "NO_LOG":
    # Try to find the actual log file created by the script
    s2, o2, e2 = c.exec_command("ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_qwen_*.log 2>/dev/null | head -1")
    logf = o2.read().decode().strip()
    if logf:
        s3, o3, e3 = c.exec_command(f"tail -15 {logf}")
        log_content = o3.read().decode().strip()
        print(f"Using log: {logf}")
print(log_content[:2000])

# GPU check
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1")
print("\nGPU0:", o.read().decode().strip())

c.close()
