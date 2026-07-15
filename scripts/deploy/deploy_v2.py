import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

# Upload files first
print("Uploading...")
sftp = c.open_sftp()
for local, remote in [
    ("eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py"),
    ("train.py", "/home/jiaqigu/mrrate_hidnet/scripts/train.py"),
    ("encoder.py", "/home/jiaqigu/mrrate_hidnet/src/encoder.py"),
]:
    sftp.put(f"C:/Users/HP/Documents/5555/{local}", remote)
    print(f"  {local}")
sftp.close()

# Install + launch in background via shell script
REMOTE = "/home/jiaqigu/mrrate_hidnet"
start_script = (
    f"cd {REMOTE} && "
    f"/home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q 2>/dev/null; "
    f"echo DEPLOY_DONE; "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"CUDA_VISIBLE_DEVICES=1 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 200 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 2000 --log_interval 10 "
    f"> outputs/report_gen/train_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)

print("Launching install + train in background...")
c.exec_command(f"bash -c '{start_script}'")

# Wait for install to finish then check training
print("Waiting for pip install + model loading (180s)...")
time.sleep(180)

stdin, stdout, stderr = c.exec_command(
    "tail -10 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_*.log | head -1)")
print(stdout.read().decode(errors="replace"))

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'")
print(f"GPU1: {stdout.read().decode().strip()}")

c.close()
