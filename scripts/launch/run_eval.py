import paramiko, io, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Upload eval runner and evaluation_v2.py
sftp = c.open_sftp()
with open(r"C:\Users\HP\Documents\5555\eval_runner.py", "rb") as f:
    sftp.putfo(io.BytesIO(f.read()), "/home/jiaqigu/mrrate_hidnet/eval_runner.py")
with open(r"C:\Users\HP\Documents\5555\evaluation_v2.py", "rb") as f:
    sftp.putfo(io.BytesIO(f.read()), "/home/jiaqigu/mrrate_hidnet/evaluation_v2.py")
sftp.close()
print("[1] eval_runner.py + evaluation_v2.py uploaded")

# Kill any existing eval
c.exec_command("pkill -9 -f eval_runner 2>/dev/null || true")
c.exec_command("tmux kill-session -t eval 2>/dev/null || true")
time.sleep(1)

# Run in tmux
c.exec_command("tmux new-session -d -s eval 'cd /home/jiaqigu/mrrate_hidnet && CUDA_VISIBLE_DEVICES=0 /home/jiaqigu/hidnet_env/bin/python -u eval_runner.py 2>&1 | tee outputs/report_gen/eval_runner.log'")
time.sleep(3)

s, o, e = c.exec_command("tmux has-session -t eval && echo OK || echo DEAD")
print("[2] Tmux:", o.read().decode().strip())

print("[3] Waiting 120s for model load + 190 sample inference...")
time.sleep(120)

s, o, e = c.exec_command("tail -30 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/eval_runner.log 2>/dev/null")
print("\n--- Runner output ---")
print(o.read().decode()[-2000:])

s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase3_val_preds.json 2>/dev/null")
print(f"\nPredictions file: {o.read().decode().strip()}")

c.close()
