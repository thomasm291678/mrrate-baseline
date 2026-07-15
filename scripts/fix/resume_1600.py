import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

PY = "/home/jiaqigu/hidnet_env/bin/python"
LOCAL = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
NAS_CKPT = "/mnt/nas1/disk07/public/qi/v3_ckpts_20260713/step_001600.pt"

# Copy step_001600.pt back from NAS
print("Copying step_001600.pt from NAS...")
stdin, o, e = c.exec_command(f"cp {NAS_CKPT} {LOCAL}/step_001600.pt && echo 'OK'", timeout=120)
print(o.read().decode().strip())

# Kill current
c.exec_command("fuser -k /dev/nvidia2 2>/dev/null; true", timeout=5)
time.sleep(8)

# Relaunch with resume
ts = time.strftime("%Y%m%d_%H%M%S")
cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"CUDA_VISIBLE_DEVICES=2 "
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
    f"--resume outputs/report_gen/step_001600.pt "
    f"> outputs/report_gen/train_v3_{ts}.log 2>&1 &"
)
c.exec_command(cmd, timeout=5)
print(f"Launched train_v3_{ts}.log --resume step_001600")

time.sleep(130)

stdin, o, e = c.exec_command(f"tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_{ts}.log", timeout=10)
print(o.read().decode())

stdin, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^2,'", timeout=10)
print(f"GPU2: {o.read().decode().strip()}")

c.close()
