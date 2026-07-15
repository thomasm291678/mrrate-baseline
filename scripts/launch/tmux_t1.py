import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
c.exec_command("tmux kill-session -t t1 2>/dev/null; true", timeout=3)
time.sleep(5)

sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"

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
    2>&1 | tee outputs/report_gen/train_v4_t1_tmux_$(date +%Y%m%d_%H%M%S).log
"""

from io import BytesIO
sftp = c.open_sftp()
sftp.putfo(BytesIO(script.encode()), "/tmp/tmux_t1.sh")
sftp.close()
c.exec_command("chmod +x /tmp/tmux_t1.sh", timeout=5)

c.exec_command("tmux new-session -d -s t1 'bash /tmp/tmux_t1.sh'", timeout=5)
print("T1 launched via tmux (session: t1)")
print("To check:  ssh jiaqigu@10.176.60.71 -t tmux attach -t t1")
print("To detach: Ctrl+B then D")

time.sleep(240)

# Check via tmux capture-pane
s, o, e = c.exec_command("tmux capture-pane -t t1 -p -S -12 2>/dev/null", timeout=10)
print("\n=== T1 output ===")
print(o.read().decode())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("tmux ls 2>/dev/null", timeout=10)
print("tmux sessions:", o.read().decode().strip())

c.close()
