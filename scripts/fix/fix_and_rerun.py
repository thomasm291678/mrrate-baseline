import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Step 1: Install defusedxml and nltk data
cmds = [
    "/home/jiaqigu/hidnet_env/bin/pip install defusedxml -q",
    "/home/jiaqigu/hidnet_env/bin/python -c 'import nltk; nltk.download(\"wordnet\", quiet=True); nltk.download(\"omw-1.4\", quiet=True)'",
]
for cmd in cmds:
    s, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode().strip())
    time.sleep(2)

# Step 2: Kill old eval tmux
c.exec_command("tmux kill-session -t eval 2>/dev/null || true")
time.sleep(1)

# Step 3: Run eval in tmux
c.exec_command("tmux new-session -d -s eval 'cd /home/jiaqigu/mrrate_hidnet && CUDA_VISIBLE_DEVICES=0 /home/jiaqigu/hidnet_env/bin/python -u eval_runner.py 2>&1 | tee outputs/report_gen/eval_runner.log'")
time.sleep(3)

s, o, e = c.exec_command("tmux has-session -t eval && echo OK || echo DEAD")
print("[Tmux]", o.read().decode().strip())

# Wait for model load + inference
print("[Wait] 120s for full inference on 190 samples...")
time.sleep(120)

# Check output
s, o, e = c.exec_command("tail -40 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/eval_runner.log 2>/dev/null")
out = o.read().decode()
print(out[-2500:])

s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase3_val_preds.json 2>/dev/null")
print("\n[Preds]", o.read().decode().strip())

s, o, e = c.exec_command("ps aux | grep eval_runner | grep -v grep | head -1")
print("[Process]", bool(o.read().decode().strip()))

c.close()
