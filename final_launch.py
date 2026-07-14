import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Install with correct python
print("Installing deps...")
c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q", timeout=10)

time.sleep(30)

# Verify
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/python -c 'import evaluate, bert_score, sklearn; print(\"ALL OK\")'", timeout=10)
print("Verify:", stdout.read().decode().strip())

# Kill failed attempt
c.exec_command("pkill -f 'train_eval_' 2>/dev/null; true")
time.sleep(2)

# Launch directly (no shell script)
print("Launching on GPU1...")
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
print(f"Command sent: eval_samples=200 on GPU1")

time.sleep(10)
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'", timeout=10)
print("GPU1:", stdout.read().decode().strip())

c.close()
