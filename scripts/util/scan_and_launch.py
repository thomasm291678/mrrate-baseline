import paramiko, time

farm04 = None
farm05 = None

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader",
            timeout=10)
        gpus = o.read().decode().strip()
        free_gpus = []
        for line in gpus.split("\n"):
            parts = [x.strip().replace(" MiB","").replace(" %","") for x in line.split(",")]
            idx, used, total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            free_mb = total - used
            is_free = used < 8000
            free_gpus.append((idx, free_mb, is_free))
            tag = "FREE" if is_free else "BUSY"
            print(f"  {name} GPU{idx}: {used}/{total}MB free={free_mb}MB util={util}% {tag}")
        
        avail = [g for g in free_gpus if g[2]]
        if name == "farm04": farm04 = avail
        if name == "farm05": farm05 = avail
        c.close()
    except Exception as ex:
        print(f"  {name}: DOWN ({ex})")
        if name == "farm04": farm04 = []
        if name == "farm05": farm05 = []

print(f"\nfarm04 free: {[g[0] for g in farm04]}")
print(f"farm05 free: {[g[0] for g in farm05]}")

# Pick best GPUs
selected = []
if farm04:
    # pick highest free mem
    best = max(farm04, key=lambda x: x[1])
    selected.append(("farm04", "10.176.60.71", best[0]))
if farm05:
    best = max(farm05, key=lambda x: x[1])
    selected.append(("farm05", "10.176.60.72", best[0]))

if not selected:
    print("NO FREE GPUS FOUND")
    exit(1)

print(f"\nSelected: {selected}")

# Launch training
PY = "/home/jiaqigu/hidnet_env/bin/python"
for svr_name, ip, gpu_id in selected:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
    
    c.exec_command(f"pkill -9 -f 'train.py' 2>/dev/null; true", timeout=5)
    time.sleep(3)
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    cmd = (
        f"cd /home/jiaqigu/mrrate_hidnet && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
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
        f"> outputs/report_gen/train_{svr_name}_gpu{gpu_id}_{ts}.log 2>&1 &"
    )
    c.exec_command(f"nohup bash -c '{cmd}' &", timeout=5)
    print(f"{svr_name}:GPU{gpu_id} launched train_{svr_name}_gpu{gpu_id}_{ts}.log")
    c.close()

print("\nWaiting for startup...")
time.sleep(150)

# Verify
for svr_name, ip, gpu_id in selected:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
    s, o, e = c.exec_command(
        f"tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_{svr_name}_gpu{gpu_id}_*.log|head -1)",
        timeout=10)
    print(f"\n{svr_name} GPU{gpu_id}:")
    print(o.read().decode().strip())
    s, o, e = c.exec_command(
        f"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^{gpu_id},'",
        timeout=10)
    print(f"GPU: {o.read().decode().strip()}")
    c.close()
