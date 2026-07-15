import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("rm -f /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt", timeout=5)
time.sleep(5)

sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"

# Sequential bash: T1 → T2 → Flair, each trains ALL params, batch=2
seq = f"""#!/bin/bash
set -e; cd /home/jiaqigu/mrrate_hidnet
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for MOD in t1 t2 flair; do
    echo "===== Training $MOD (all params) ====="
    {PY} -u scripts/train_v4.py \\
        --modality "$MOD" --augment \\
        --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
        --projector attn \\
        --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
        --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
        --batch_size 2 --ga_steps 4 --epochs 3 --num_workers 2 \\
        --lr 1e-04 --cnn_lr 1e-05 --grid 2 \\
        --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
        --use_amp --eval_samples 100 \\
        --log_dir outputs/report_gen \\
        --save_interval 200 --auto_save_interval 100 --log_interval 10 \\
        2>&1 | tee "outputs/report_gen/train_v4_$MOD.log"
    echo "$MOD DONE"
done
echo "ALL THREE DONE"
"""

sftp = c.open_sftp()
from io import BytesIO
sftp.putfo(BytesIO(seq.encode()), "/tmp/v4_all3.sh")
sftp.close()
c.exec_command("chmod +x /tmp/v4_all3.sh", timeout=5)

ts = time.strftime("%Y%m%d_%H%M%S")
c.exec_command(
    f"nohup bash /tmp/v4_all3.sh > outputs/report_gen/v4_all3_{ts}.log 2>&1 &",
    timeout=5)
print(f"Launched: v4_all3_{ts}.log")
print("T1 → T2 → Flair | Each: 1 encoder, train ALL params, batch=2 ga=4 (eff=8)")

time.sleep(240)

# Check T1 log
s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1.log 2>/dev/null | head -1",
    timeout=10)
t1_log = o.read().decode().strip()
if t1_log:
    s, o, e = c.exec_command(f"tail -12 {t1_log}", timeout=10)
    print("\n=== T1 Training ===")
    print(o.read().decode())
else:
    print("\nT1 still loading...")

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("ps -u jiaqigu | grep -E 'train_v4|bash.*train' | grep -v grep | wc -l", timeout=10)
print(f"Alive: {o.read().decode().strip()}")

c.close()
