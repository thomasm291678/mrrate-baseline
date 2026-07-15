import paramiko, io, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Upload all 5 files
sftp = c.open_sftp()
files_local = [
    (r"C:\Users\HP\Documents\5555\train_v5_phase3.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase3.py"),
    (r"C:\Users\HP\Documents\5555\train_v5_phase1.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase1.py"),
    (r"C:\Users\HP\Documents\5555\train_v5_phase2.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase2.py"),
    (r"C:\Users\HP\Documents\5555\server_code\mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py"),
    (r"C:\Users\HP\Documents\5555\encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py"),
]
for local, remote in files_local:
    with open(local, "rb") as f:
        sftp.putfo(io.BytesIO(f.read()), remote)
    print(f"Uploaded: {local.split(chr(92))[-1]}")
sftp.close()

# Kill old
c.exec_command("pkill -9 -f train_v5_phase3 2>/dev/null || true")
c.exec_command("tmux kill-session -t phase3 2>/dev/null || true")
time.sleep(2)

# Start tmux
launch_path = "/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3_opt.sh"
c.exec_command(f"tmux new-session -d -s phase3 '{launch_path} 2>&1 | tee /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_opt.log'")
time.sleep(3)
s, o, e = c.exec_command("tmux has-session -t phase3 && echo SESSION_OK || echo NO_SESSION")
print("Tmux:", o.read().decode().strip())

print("Waiting 90s for compile + data loading...")
time.sleep(90)

s, o, e = c.exec_command("ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_qwen_*.log 2>/dev/null | head -1")
logf = o.read().decode().strip()
if logf:
    s2, o2, e2 = c.exec_command(f"tail -15 {logf}")
    print("\n--- Training log ---")
    print(o2.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1")
print(f"\nGPU0: {o.read().decode().strip()}")

c.close()
