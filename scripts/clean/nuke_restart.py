import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Find and kill ALL GPU processes via nvidia-smi
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,gpu_name --format=csv,noheader 2>/dev/null", timeout=10)
print("GPU procs:", o.read().decode().strip())

# Kill all python/GPU processes
c.exec_command("pkill -9 -f 'python.*train.py' 2>/dev/null; true", timeout=5)
time.sleep(10)

# Verify
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^0,|^2,'; "
    "ps -u jiaqigu | grep python | grep -v grep | grep -v '/usr/'", timeout=10)
print("After kill:", o.read().decode().strip())

# Launch GPU0
ts = time.strftime("%Y%m%d_%H%M%S")
cmd0 = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 200 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --log_interval 10 "
    f"> outputs/report_gen/train_gpu0_{ts}.log 2>&1 &",
    timeout=5)
print("GPU0 launched")

time.sleep(10)

# Launch GPU2  
ts2 = time.strftime("%Y%m%d_%H%M%S")
cmd2 = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 200 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --log_interval 10 "
    f"> outputs/report_gen/train_gpu2_{ts2}.log 2>&1 &",
    timeout=5)
print("GPU2 launched")

time.sleep(160)

s, o, e = c.exec_command(
    f"echo '=== GPU0 ==='; tail -3 outputs/report_gen/train_gpu0_{ts}.log; "
    f"echo '=== GPU2 ==='; tail -3 outputs/report_gen/train_gpu2_{ts2}.log; "
    f"echo '=== GPU ==='; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^0,|^2,'",
    timeout=10)
print(o.read().decode())

c.close()
