import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Clean
c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
time.sleep(5)

# Upload
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Write sequential bash script
script = f"""#!/bin/bash
set -e
cd /home/jiaqigu/mrrate_hidnet
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== Phase 1: T1 ====="
{PY} -u scripts/train_v4.py \\
    --modality t1 --phase uniformer --augment --auto_resume \\
    --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
    --projector attn \\
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
    --batch_size 4 --ga_steps 2 --epochs 3 --num_workers 2 \\
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \\
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
    --use_amp --eval_samples 100 \\
    --log_dir outputs/report_gen \\
    --save_interval 200 --auto_save_interval 100 --log_interval 10 \\
    2>&1 | tee outputs/report_gen/train_v4_t1_$(date +%Y%m%d_%H%M%S).log
echo "T1 DONE"

echo "===== Phase 2: T2 ====="
{PY} -u scripts/train_v4.py \\
    --modality t2 --phase uniformer --augment --auto_resume \\
    --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
    --projector attn \\
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
    --batch_size 4 --ga_steps 2 --epochs 3 --num_workers 2 \\
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \\
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
    --use_amp --eval_samples 100 \\
    --log_dir outputs/report_gen \\
    --save_interval 200 --auto_save_interval 100 --log_interval 10 \\
    2>&1 | tee outputs/report_gen/train_v4_t2_$(date +%Y%m%d_%H%M%S).log
echo "T2 DONE"

echo "===== Phase 3: Flair ====="
{PY} -u scripts/train_v4.py \\
    --modality flair --phase uniformer --augment --auto_resume \\
    --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
    --projector attn \\
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
    --batch_size 4 --ga_steps 2 --epochs 3 --num_workers 2 \\
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \\
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
    --use_amp --eval_samples 100 \\
    --log_dir outputs/report_gen \\
    --save_interval 200 --auto_save_interval 100 --log_interval 10 \\
    2>&1 | tee outputs/report_gen/train_v4_flair_$(date +%Y%m%d_%H%M%S).log
echo "ALL DONE"
"""

sftp = c.open_sftp()
sftp.putfo(script.encode(), "/tmp/v4_sequential.sh")
sftp.close()
c.exec_command("chmod +x /tmp/v4_sequential.sh", timeout=5)

ts = time.strftime("%Y%m%d_%H%M%S")
c.exec_command(
    f"nohup bash /tmp/v4_sequential.sh > outputs/report_gen/v4_sequential_{ts}.log 2>&1 &",
    timeout=5)
print(f"Launched sequential pipeline: v4_sequential_{ts}.log")
print("T1(batch=4) → T2(batch=4) → Flair(batch=4) | Each: 3 epochs, eff_batch=8")

time.sleep(240)

s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_*.log 2>/dev/null | head -1",
    timeout=10)
log = o.read().decode().strip()

if log:
    s, o, e = c.exec_command(f"tail -12 {log}", timeout=10)
    print("\n=== T1 log ===")
    print(o.read().decode())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep | wc -l", timeout=10)
print(f"Processes alive: {o.read().decode().strip()}")

c.close()
