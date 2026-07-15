import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Kill everything first
print("Killing processes...")
c.exec_command("pkill -9 -f 'train.py' 2>/dev/null; pkill -9 -f 'watchdog.sh' 2>/dev/null; true")
time.sleep(3)

# Verify nothing running
stdin, stdout, stderr = c.exec_command("ps -u jiaqigu | grep python")
print("Python procs:", stdout.read().decode().strip() or "none")

# Verify GPU free
stdin, stdout, stderr = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
print("GPU6:", stdout.read().decode().strip())

# Verify fix is in place
stdin, stdout, stderr = c.exec_command("grep -c 'Resizing.*ckpt.*tokenizer' /home/jiaqigu/mrrate_hidnet/scripts/train.py")
count = stdout.read().decode().strip()
print(f"Fix lines found: {count}")

# Direct start (no watchdog)
print("\nStarting training directly...")
c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=6 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--v1_ckpt outputs/report_gen/best_model.pt "
    "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    "--batch_size 2 --ga_steps 2 --epochs 5 "
    "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    "--vit_dim 512 --vit_heads 8 --vit_depth 2 --use_amp "
    "--log_dir outputs/report_gen "
    "> outputs/report_gen/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)

print("Training started. Waiting 90s for model loading...")
time.sleep(90)

# Check status
stdin, stdout, stderr = c.exec_command(
    "tail -25 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print("\nLog:\n", stdout.read().decode(errors="replace"))

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
print("GPU6:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command("ps -u jiaqigu | grep python")
print("Python:", stdout.read().decode().strip())

c.close()
