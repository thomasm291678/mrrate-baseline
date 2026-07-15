import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Write a bash script that runs everything
script = """#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
export CUDA_VISIBLE_DEVICES=0
/home/jiaqigu/hidnet_env/bin/python -u eval_runner.py
"""

sftp = c.open_sftp()
import io
sftp.putfo(io.BytesIO(script.encode()), "/home/jiaqigu/mrrate_hidnet/run_eval.sh")
sftp.close()
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/run_eval.sh")

c.exec_command("tmux kill-session -t eval 2>/dev/null || true")
time.sleep(1)

c.exec_command("tmux new-session -d -s eval /home/jiaqigu/mrrate_hidnet/run_eval.sh")
time.sleep(3)

s, o, e = c.exec_command("tmux has-session -t eval && echo SESSION_OK || echo NO_SESSION")
print("Tmux:", o.read().decode().strip())

print("Waiting 180s for full inference...")
time.sleep(180)

# Check output from tmux pane
s, o, e = c.exec_command("tmux capture-pane -t eval -p -S -100 2>/dev/null")
print("\n--- Eval output ---")
print(o.read().decode()[-3000:])

# Check preds file
s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase3_val_preds.json 2>/dev/null")
preds_file = o.read().decode().strip()
print(f"\nPredictions: {preds_file}")

if preds_file:
    s, o, e = c.exec_command("wc -l /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase3_val_preds.json")
    print(o.read().decode().strip())

s, o, e = c.exec_command("ps aux | grep eval_runner | grep -v grep | head -1")
print(f"Running: {bool(o.read().decode().strip())}")

c.close()
