import paramiko, time, os

FARM01 = {"hostname": "10.176.60.71", "username": "jiaqigu", "password": "lijia7272"}
LOCAL = r"C:\Users\HP\Documents\5555"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(**FARM01, timeout=30)

print("1. Killing current training...")
c.exec_command("pkill -9 -f train.py 2>/dev/null; true")
c.exec_command("pkill -9 -f watchdog 2>/dev/null; true")
time.sleep(3)

stdin, stdout, stderr = c.exec_command("ps -u jiaqigu | grep python || echo 'no python'")
print("   Python:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
print("   GPU6:", stdout.read().decode().strip())

print("\n2. Uploading new code...")
c.exec_command(f"mkdir -p {REMOTE}/server_code")
sftp = c.open_sftp()

for local_rel, remote in [
    ("train.py", f"{REMOTE}/scripts/train.py"),
    ("run.sh", f"{REMOTE}/run.sh"),
    ("encoder.py", f"{REMOTE}/src/encoder.py"),
    ("server_code/densenet3d.py", f"{REMOTE}/src/densenet3d.py"),
    ("server_code/mrrate_dataset.py", f"{REMOTE}/server_code/mrrate_dataset.py"),
]:
    lp = os.path.join(LOCAL, local_rel)
    sftp.put(lp, remote)
    print(f"   {local_rel} ({os.path.getsize(lp)/1024:.0f}KB)")

sftp.close()
c.exec_command(f"chmod +x {REMOTE}/run.sh")
print("   Done.")

print("\n3. Starting training directly (batch_size=4, workers=4)...")
c.exec_command(
    f"cd {REMOTE} && "
    f"CUDA_VISIBLE_DEVICES=6 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 4 --ga_steps 1 --epochs 5 "
    f"--num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --log_dir outputs/report_gen "
    f"--save_interval 1000 --log_interval 10 "
    f"> outputs/report_gen/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)

log_before = sftp.listdir if hasattr(sftp, 'listdir') else None
c.exec_command("sleep 3")

print("   Waiting 120s for model loading + first steps...")
time.sleep(120)

stdin, stdout, stderr = c.exec_command(
    "tail -20 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print("\n4. Latest log:")
print(stdout.read().decode(errors="replace"))

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^6'")
print("   GPU6:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command("ps -u jiaqigu | grep python")
print("   Python:", stdout.read().decode().strip())

c.close()
print("\nDone.")
