import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill GPU3 zombie + our stuff
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
time.sleep(3)

# Check ALL GPUs
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader", timeout=10)
print("All GPUs:")
for line in o.read().decode().strip().split('\n'):
    idx, used, free = [x.strip() for x in line.split(',')]
    used_mb = int(used.replace(' MiB',''))
    if used_mb < 5000:
        print(f"  GPU{idx}: used={used} free={free}  <-- FREE")
    else:
        print(f"  GPU{idx}: used={used} free={free}")

# Pick first free GPU
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -v '%' | awk -F',' '{split($2,a,\" \"); if (a[1]+0 < 5000) print $1}'", timeout=10)
free_gpus = [x.strip() for x in o.read().decode().strip().split('\n') if x.strip()]
print(f"\nFree GPUs: {free_gpus}")

if free_gpus:
    gpu_id = free_gpus[0]
    PY = "/home/jiaqigu/hidnet_env/bin/python"
    ts = time.strftime("%Y%m%d_%H%M%S")
    cmd = (
        f"cd /home/jiaqigu/mrrate_hidnet && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"nohup {PY} -u scripts/train_v5.py "
        f"--phase encoder --modality all --augment "
        f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
        f"--batch_size 16 --ga_steps 1 --epochs 5 --num_workers 4 "
        f"--lr 3e-4 --wd 1e-4 --grad_clip 1.0 "
        f"--grid 2 --base_ch 32 "
        f"--batch_id batch27 "
        f"--log_dir outputs/report_gen "
        f"--save_interval 100 --auto_save_interval 50 --log_interval 5 "
        f">> outputs/report_gen/train_v5_b27_{ts}.log 2>&1 &"
    )
    c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
    print(f"\nLAUNCHED on GPU{gpu_id}: train_v5_b27_{ts}.log")
    
    time.sleep(150)
    s, o, e = c.exec_command(f"tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_{ts}.log", timeout=10)
    print("\n" + o.read().decode())
    
    s, o, e = c.exec_command(f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^{gpu_id},'", timeout=10)
    print(f"GPU{gpu_id}: {o.read().decode().strip()}")

c.close()
