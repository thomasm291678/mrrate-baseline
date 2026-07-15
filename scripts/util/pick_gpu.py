import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader", timeout=10)
best_gpu, best_free = None, 0
for line in o.read().decode().strip().split('\n'):
    idx, free = [x.strip() for x in line.split(',')]
    free_mb = int(free.replace(' MiB',''))
    if free_mb > best_free:
        best_free, best_gpu = free_mb, idx
    print(f"  GPU{idx}: free={free} {'<-- pick' if free_mb == best_free else ''}")

print(f"\nPicking GPU{best_gpu} ({best_free}MiB free)")

if best_gpu:
    PY = "/home/jiaqigu/hidnet_env/bin/python"
    ts = time.strftime("%Y%m%d_%H%M%S")
    cmd = (
        f"cd /home/jiaqigu/mrrate_hidnet && "
        f"CUDA_VISIBLE_DEVICES={best_gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
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
    print(f"Launched on GPU{best_gpu}")

    time.sleep(150)
    s, o, e = c.exec_command(f"tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_{ts}.log 2>/dev/null", timeout=10)
    print("\n" + o.read().decode())
    s, o, e = c.exec_command(f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^{best_gpu},'", timeout=10)
    print(f"GPU{best_gpu}: {o.read().decode().strip()}")
    s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
    print(f"Procs: {o.read().decode().strip()}")

c.close()
