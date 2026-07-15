import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"

# List GPU0 processes
s, o, e = c.exec_command("fuser -v /dev/nvidia0 2>&1; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null", timeout=10)
print("GPU0 procs:", o.read().decode().strip())

# Try kill all on GPU0
c.exec_command("fuser -k /dev/nvidia0 2>/dev/null; true", timeout=5)
time.sleep(5)

# Alternative: reset GPU0 via nvidia-smi
c.exec_command("nvidia-smi -i 0 -r 2>/dev/null; true", timeout=5)
time.sleep(10)

# Verify
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^0,'", timeout=10)
print("GPU0 after reset:", o.read().decode().strip())

# If still dirty, try killing ALL python3 processes
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^0,'", timeout=10)
mem = o.read().decode().strip()
if "37270" in mem or "37322" in mem or "37000" in mem:
    print("Still occupied, killing all python...")
    c.exec_command("pkill -9 python 2>/dev/null; pkill -9 python3 2>/dev/null; true", timeout=5)
    time.sleep(10)
    s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^0,'", timeout=10)
    print("GPU0 after pkill:", o.read().decode().strip())

# Launch GPU0 from step_001200 checkpoint
ts = time.strftime("%Y%m%d_%H%M%S")
launch = (
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "nohup " + PY + " -u scripts/train.py "
    "--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    "--v1_ckpt outputs/report_gen/best_model.pt "
    "--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    "--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
    "--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    "--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    "--use_amp --eval_samples 200 "
    "--log_dir outputs/report_gen "
    "--save_interval 200 --log_interval 10 "
    "--resume outputs/report_gen/step_001200.pt "
    "> outputs/report_gen/train_gpu0_" + ts + ".log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{launch}' &", timeout=5)
print("GPU0 launched: train_gpu0_" + ts)

time.sleep(160)

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^0,|^2,'", timeout=10)
print("GPUs:", o.read().decode().strip())

c.close()
