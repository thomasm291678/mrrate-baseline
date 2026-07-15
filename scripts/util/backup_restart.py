import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

NAS = "/mnt/nas1/disk07"
SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
PY = "/home/jiaqigu/hidnet_env/bin/python"

# Check NAS space
stdin, o, e = c.exec_command(f"df -h {NAS}", timeout=10)
print("NAS:", o.read().decode().strip().split("\n")[-1])

# Create nas dir
c.exec_command(f"mkdir -p {NAS}/jiaqigu_mrrate_ckpts", timeout=10)

# Copy all step checkpoints + best_model
stdin, o, e = c.exec_command(
    f"cp -v {SRC}/step_*.pt {NAS}/jiaqigu_mrrate_ckpts/ 2>&1 && "
    f"cp -v {SRC}/best_model.pt {NAS}/jiaqigu_mrrate_ckpts/ 2>&1",
    timeout=300)
print("Copy:", o.read().decode().strip()[:500])

# Verify
stdin, o, e = c.exec_command(f"ls -lh {NAS}/jiaqigu_mrrate_ckpts/", timeout=10)
print("NAS ckpts:", o.read().decode().strip())

# Clean local: keep only last 2 step checkpoints + best_model
stdin, o, e = c.exec_command(f"ls -t {SRC}/step_*.pt 2>/dev/null", timeout=10)
keep = o.read().decode().strip().split("\n")[:2]
print(f"Keeping: {keep}")

c.exec_command(
    f"cd {SRC} && "
    f"ls -t step_*.pt | tail -n +3 | xargs rm -f 2>/dev/null; "
    f"rm -f step_001800.pt train_eval_*.log train_farm02_*.log train_202*.log "
    f"watch_*.log fix*.log repair.log startup.log watchdog*.log monitor.log 2>/dev/null; "
    f"true", timeout=10)

stdin, o, e = c.exec_command(f"df -h {SRC}/..", timeout=10)
print("Disk after:", o.read().decode().strip().split("\n")[-1])

# Relaunch
time.sleep(3)
ts = time.strftime("%Y%m%d_%H%M%S")
log_name = f"train_v3_{ts}.log"
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
    f"> outputs/report_gen/{log_name} 2>&1 &"
)
c.exec_command(cmd, timeout=5)
print(f"Launched {log_name}")

time.sleep(130)
stdin, o, e = c.exec_command(f"tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/{log_name}", timeout=10)
print(o.read().decode())
stdin, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^2,'", timeout=10)
print(f"GPU2: {o.read().decode().strip()}")

c.close()
