import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("pkill -9 -f 'train_v5' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("rm -f /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt", timeout=5)
time.sleep(8)

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")

cmd = (
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"nohup {PY} -u scripts/train_v5.py "
    f"--phase encoder --modality all --augment "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--batch_size 16 --ga_steps 1 --epochs 2 --num_workers 4 "
    f"--lr 3e-4 --wd 1e-4 --grad_clip 1.0 "
    f"--grid 2 --base_ch 32 "
    f"--log_dir outputs/report_gen "
    f"--save_interval 1000 --auto_save_interval 200 --log_interval 10 "
    f">> outputs/report_gen/train_v5_encoder_{ts}.log 2>&1 &"
)
c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
print(f"Fresh start: train_v5_encoder_{ts}.log (workers=4, batch=16, CLEAN)")

time.sleep(180)

s, o, e = c.exec_command(
    f"tail -10 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_{ts}.log 2>/dev/null",
    timeout=10)
print("\n" + o.read().decode())

for i in range(2):
    s, o, e = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
    print(f"GPU3 [{i}]: {o.read().decode().strip()}")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

c.close()
