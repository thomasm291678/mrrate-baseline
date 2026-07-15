import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill ALL our processes
c.exec_command("pkill -9 -u jiaqigu -f 'train_v4' 2>/dev/null; true", timeout=5)
c.exec_command("pkill -9 -u jiaqigu -f 'train_v5' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia7 2>/dev/null; true", timeout=5)
c.exec_command("tmux kill-server 2>/dev/null; true", timeout=3)
time.sleep(8)

# Verify clean
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v' | grep -v grep", timeout=10)
procs = o.read().decode().strip()
print("Processes:", procs if procs else "(none)")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^(3|7),'", timeout=10)
print(o.read().decode())

# Upload latest files
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
sftp.put(r"C:\Users\HP\Documents\5555\train_v5.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5.py")
sftp.put(r"C:\Users\HP\Documents\5555\server_code\mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")

# Phase 1: Contrastive encoder training on GPU3 (all 3 mods, no Qwen)
cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v5.py "
    f"--phase encoder --modality all --augment "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--batch_size 4 --ga_steps 2 --epochs 2 --num_workers 2 "
    f"--lr 3e-4 --wd 1e-4 --grad_clip 1.0 "
    f"--grid 2 --base_ch 32 "
    f"--use_amp "
    f"--log_dir outputs/report_gen "
    f"--save_interval 1000 --auto_save_interval 200 --log_interval 10 "
    f">> outputs/report_gen/train_v5_encoder_{ts}.log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"\nV5 Phase1 (encoder) launched: train_v5_encoder_{ts}.log")
print(f"  GPU3 | batch=4 ga=2 eff=8 | 2 epochs | 89k samples | no Qwen")

time.sleep(210)

# Check
s, o, e = c.exec_command(
    f"tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_{ts}.log 2>/dev/null",
    timeout=10)
print("\n" + o.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Alive: {o.read().decode().strip()}")

c.close()
