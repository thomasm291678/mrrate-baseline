import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

# Kill the failed GPU0 process if any
c.exec_command("pkill -f 'CUDA_VISIBLE_DEVICES=0.*train.py' 2>/dev/null; true")
time.sleep(2)

# Install deps
print("1. Installing eval deps...")
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q 2>&1 | tail -3",
    timeout=120)
print(stdout.read().decode(errors="replace") or stderr.read().decode(errors="replace"))

# Upload new files
print("2. Uploading...")
sftp = c.open_sftp()
for local, remote in [
    ("eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py"),
    ("train.py", "/home/jiaqigu/mrrate_hidnet/scripts/train.py"),
    ("encoder.py", "/home/jiaqigu/mrrate_hidnet/src/encoder.py"),
]:
    sftp.put(f"C:/Users/HP/Documents/5555/{local}", remote)
    print(f"  {local}")
sftp.close()

# Launch on GPU1
print("3. Launching on farm04 GPU1...")
REMOTE = "/home/jiaqigu/mrrate_hidnet"
cmd = (
    f"cd {REMOTE} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
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
c.exec_command(cmd)

print("4. Waiting 130s for warmup...")
time.sleep(130)

stdin, stdout, stderr = c.exec_command(
    "tail -10 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_*.log | head -1)")
out = stdout.read().decode(errors="replace")
print(out)

if "loss=" in out:
    print("OK - training started with eval!")
else:
    print("WARN - check log manually")

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'")
print("GPU1:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command(
    "ps -u jiaqigu | grep train.py | grep -v grep | awk '{print $2}'")
pids = stdout.read().decode().strip()
print(f"train.py PIDs: {pids}")

c.close()
