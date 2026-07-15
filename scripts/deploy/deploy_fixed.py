import paramiko, os, time

FARM01 = {"hostname": "10.176.60.71", "username": "jiaqigu", "password": "lijia7272"}
LOCAL = r"C:\Users\HP\Documents\5555"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(**FARM01, timeout=30)

c.exec_command("pkill -9 -f train.py 2>/dev/null; true")
time.sleep(3)

sftp = c.open_sftp()
files = [
    ("encoder.py", f"{REMOTE}/src/encoder.py"),
    ("train.py", f"{REMOTE}/scripts/train.py"),
    ("run.sh", f"{REMOTE}/run.sh"),
    ("server_code/densenet3d.py", f"{REMOTE}/src/densenet3d.py"),
    ("server_code/mrrate_dataset.py", f"{REMOTE}/server_code/mrrate_dataset.py"),
]
for local_rel, remote in files:
    lp = os.path.join(LOCAL, local_rel)
    sftp.put(lp, remote)
    print(f"Upload: {local_rel}")
sftp.close()

cmd = (
    f"cd {REMOTE} && CUDA_VISIBLE_DEVICES=6 nohup /home/jiaqigu/hidnet_env/bin/python -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 6 --ga_steps 1 --epochs 5 --num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --compile "
    f"--log_dir outputs/report_gen "
    f"--save_interval 2000 --log_interval 10 "
    f"> outputs/report_gen/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)
print("Starting...")
c.exec_command(cmd)

print("Wait 200s (compile + load)...")
time.sleep(200)

cmds = [
    'echo "=== LOG ==="',
    'tail -15 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)',
    'echo "=== GPU6 ==="',
    'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep "^6"',
    'echo "=== PY ==="',
    'ps -u jiaqigu | grep python',
]
stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
print(stdout.read().decode(errors="replace"))
c.close()
