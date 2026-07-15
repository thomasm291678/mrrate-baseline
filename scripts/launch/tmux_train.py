import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Clean leaked semaphores
c.exec_command(
    "for s in $(ipcs -s | grep jiaqigu | awk '{print $2}'); do ipcrm -s $s 2>/dev/null; done; "
    "echo 'semaphores cleaned'",
    timeout=10)

# Kill any existing tmux session named v4
c.exec_command("tmux kill-session -t v4 2>/dev/null; true", timeout=5)
time.sleep(2)

# Upload latest train_v4.py
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()

PY = "/home/jiaqigu/hidnet_env/bin/python"
ts = time.strftime("%Y%m%d_%H%M%S")
LOG = f"outputs/report_gen/train_v4_gpu3_{ts}.log"

script = f"""#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{PY} -u scripts/train_v4.py \\
    --brainmvp_ckpt outputs/report_gen/BrainMVP_uniformer.pt \\
    --projector attn --augment --auto_resume \\
    --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \\
    --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \\
    --batch_size 1 --ga_steps 5 --epochs 5 --num_workers 2 \\
    --lr 1e-4 --cnn_lr 1e-5 --grid 2 \\
    --vit_dim 512 --vit_heads 8 --vit_depth 2 \\
    --use_amp --eval_samples 100 \\
    --log_dir outputs/report_gen \\
    --save_interval 200 --auto_save_interval 50 --log_interval 10 \\
    2>&1 | tee {LOG}
"""

# Write script to server
c.exec_command(f"cat > /tmp/v4_train.sh << 'TMUXEOF'\n{script}\nTMUXEOF", timeout=10)
c.exec_command("chmod +x /tmp/v4_train.sh", timeout=5)

# Launch in detached tmux with setsid
c.exec_command(
    "tmux new-session -d -s v4 'bash /tmp/v4_train.sh'",
    timeout=5)
print("Launched in tmux session 'v4'")

time.sleep(8)

# Verify
s, o, e = c.exec_command("tmux ls 2>/dev/null", timeout=10)
print(f"tmux sessions: {o.read().decode().strip()}")

s, o, e = c.exec_command(
    "ps -u jiaqigu | grep 'train_v4' | grep -v grep | awk '{print $2}'",
    timeout=10)
pid = o.read().decode().strip()
print(f"V4 PID: {pid}")

time.sleep(3)
s, o, e = c.exec_command(
    f"cat /proc/{pid}/status 2>/dev/null | grep -E 'PPid|State|Name'", timeout=10)
print(o.read().decode().strip())

print(f"\nView tmux: ssh jiaqigu@10.176.60.71 -t tmux attach -t v4")
print(f"Detach: Ctrl+B then D")
print(f"Kill: tmux kill-session -t v4")

c.close()
