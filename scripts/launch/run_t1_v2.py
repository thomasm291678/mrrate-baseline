import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("kill -9 3895976 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("tmux kill-session -t t1 2>/dev/null; true", timeout=3)
time.sleep(8)

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")

script = f"""#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{PY} -u scripts/train_v4.py \\
    --modality t1 --augment \\
    --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
    --projector attn \\
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
    --batch_size 2 --ga_steps 4 --epochs 3 --num_workers 2 \\
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \\
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
    --use_amp --eval_samples 100 \\
    --log_dir outputs/report_gen \\
    --save_interval 200 --auto_save_interval 100 --log_interval 10 \\
    >> outputs/report_gen/train_v4_t1_{ts}.log 2>&1
"""

from io import BytesIO
sftp = c.open_sftp()
sftp.putfo(BytesIO(script.encode()), "/tmp/tmux_t1_v2.sh")
sftp.close()
c.exec_command("chmod +x /tmp/tmux_t1_v2.sh", timeout=5)

c.exec_command("tmux new-session -d -s t1 'bash /tmp/tmux_t1_v2.sh'", timeout=5)
print(f"T1 v2 started: train_v4_t1_{ts}.log (NO tee — direct append, no pipe blocking)")

time.sleep(240)

# Check
s, o, e = c.exec_command(
    f"tail -8 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_{ts}.log 2>/dev/null",
    timeout=10)
print("\n=== Status ===")
print(o.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

c.close()
