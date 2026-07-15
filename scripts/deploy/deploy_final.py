import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Install evaluate with explicit pip path
PY = "/home/jiaqigu/hidnet_env/bin/python"
PIP = "/home/jiaqigu/hidnet_env/bin/pip"

# Step 1: pip install (long timeout)
print("pip installing evaluate...")
stdin, stdout, stderr = c.exec_command(
    f"{PIP} install evaluate bert_score scikit-learn rouge_score 2>&1", timeout=120)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(out[-200:] if len(out) > 200 else out)
print(err[-200:] if len(err) > 200 else err)

# Step 2: verify
print("\nVerifying...")
stdin, stdout, stderr = c.exec_command(
    f"{PY} -c 'import evaluate, bert_score, sklearn; print(\"ALL MODULES OK\")'", timeout=15)
ver = stdout.read().decode().strip()
err2 = stderr.read().decode().strip()
print(f"stdout: {ver}")
print(f"stderr: {err2[:200] if err2 else 'none'}")

if "ALL MODULES OK" not in ver:
    print("INSTALL FAILED!")
    c.close()
    exit(1)

# Step 3: kill old and launch
print("\nLaunching training...")
c.exec_command("pkill -f 'train_eval_' 2>/dev/null; true", timeout=5)
time.sleep(3)

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

print(f"Launched. Log: train_eval_{ts}.log")
print("Waiting 120s for model loading...")
time.sleep(120)

# Step 4: verify
stdin, stdout, stderr = c.exec_command(
    f"tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_{ts}.log", timeout=10)
print("Training log:", stdout.read().decode(errors="replace").strip()[:500])

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'", timeout=10)
print("GPU1:", stdout.read().decode().strip())

c.close()
