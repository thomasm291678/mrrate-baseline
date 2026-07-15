import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"
PIP = "/home/jiaqigu/hidnet_env/bin/pip"

# Install only evaluate, scikit-learn, rouge_score (skip bert_score)
print("pip install (skip bert_score)...")
stdin, stdout, stderr = c.exec_command(
    f"{PIP} install evaluate scikit-learn rouge_score 2>&1 | tail -5", timeout=60)
print(stdout.read().decode(errors="replace"))

# Verify
stdin, stdout, stderr = c.exec_command(
    f"{PY} -c 'import evaluate, sklearn; print(\"OK\")'", timeout=15)
print("Verify:", stdout.read().decode().strip() + stderr.read().decode().strip()[:200])

# Upload updated eval_report.py
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py")
sftp.close()
print("Uploaded eval_report.py")

# Kill old
c.exec_command("pkill -f 'train_eval_' 2>/dev/null; true")
time.sleep(3)

# Launch
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"CUDA_VISIBLE_DEVICES=1 "
    f"nohup {PY} -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 200 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 2000 --log_interval 10 "
    f"> outputs/report_gen/train_eval_{ts}.log 2>&1 &"
)
c.exec_command(cmd, timeout=5)
print(f"Launched train_eval_{ts}.log")

# Wait and check
time.sleep(120)
stdin, stdout, stderr = c.exec_command(
    f"tail -8 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_{ts}.log", timeout=10)
print("Train:", stdout.read().decode(errors="replace").strip()[:500])

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'", timeout=10)
print("GPU1:", stdout.read().decode().strip())

c.close()
