import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill everything
c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("tmux kill-session -t t1 2>/dev/null; true", timeout=3)
time.sleep(8)

# Upload
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")
LOG = f"outputs/report_gen/train_v4_t1_{ts}.log"

cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v4.py "
    f"--modality t1 --augment "
    f"--brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt "
    f"--projector attn "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 2 --ga_steps 4 --epochs 3 --num_workers 2 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --eval_samples 100 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 200 --auto_save_interval 100 --log_interval 10 "
    f">> {LOG} 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"T1 deployed (nohup, no tee, no fsync): {LOG}")

time.sleep(240)

s, o, e = c.exec_command(f"tail -8 /home/jiaqigu/mrrate_hidnet/{LOG} 2>/dev/null", timeout=10)
print("\n=== Status ===")
print(o.read().decode())

s, o, e = c.exec_command("pgrep -f 'train_v4.*t1' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

# Double check after another 120s
time.sleep(120)

s, o, e = c.exec_command(f"tail -5 /home/jiaqigu/mrrate_hidnet/{LOG} 2>/dev/null", timeout=10)
print(f"\n=== 120s later ===")
print(o.read().decode().strip())

s, o, e = c.exec_command("pgrep -f 'train_v4.*t1' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

c.close()
