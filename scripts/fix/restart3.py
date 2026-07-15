import paramiko, io, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

with open(r"C:\Users\HP\Documents\5555\train_v5_phase3.py", "rb") as f:
    data = f.read()

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(data), "/home/jiaqigu/mrrate_hidnet/scripts/train_v5_phase3.py")
sftp.close()
print("[1] Fixed phase3 uploaded (no torch.compile)")

c.exec_command("pkill -9 -f train_v5_phase3 2>/dev/null || true")
c.exec_command("tmux kill-session -t phase3 2>/dev/null || true")
c.exec_command("rm -f /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase3_latest.pt")
time.sleep(1)

c.exec_command("tmux new-session -d -s phase3 '/home/jiaqigu/mrrate_hidnet/scripts/launch_phase3_opt.sh 2>&1 | tee /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_opt3.log'")
time.sleep(3)

s, o, e = c.exec_command("tmux has-session -t phase3 && echo OK || echo DEAD")
print("[2] Tmux:", o.read().decode().strip())

print("[3] Waiting 90s for loading...")
time.sleep(90)

s, o, e = c.exec_command("ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_phase3_qwen_*.log 2>/dev/null | head -1")
logf = o.read().decode().strip()
if logf:
    s2, o2, e2 = c.exec_command(f"tail -10 {logf}")
    print("\n--- Log ---")
    print(o2.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -1")
print(f"\nGPU0: {o.read().decode().strip()}")

s, o, e = c.exec_command("ps aux | grep train_v5_phase3 | grep -v grep | head -1")
print(f"Running: {bool(o.read().decode().strip())}")

c.close()
