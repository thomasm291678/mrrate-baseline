import paramiko, time

time.sleep(60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Find latest T1 log
s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_1*.log | head -1",
    timeout=10)
log = o.read().decode().strip()
print(f"Log: {log}")

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print("Last 5 lines:")
print(o.read().decode().strip())

s, o, e = c.exec_command(f"wc -l {log}", timeout=10)
print(f"\nLines: {o.read().decode().strip()}")

s, o, e = c.exec_command(f"ls -la {log}", timeout=10)
print(f"Size: {o.read().decode().strip()}")

# Process alive?
s, o, e = c.exec_command("pgrep -f 'train_v4.py.*t1' | head -3", timeout=10)
pids = o.read().decode().strip()
if pids:
    for pid in pids.split():
        s2, o2, e2 = c.exec_command(f"cat /proc/{pid}/status 2>/dev/null | grep -E 'State|VmRSS|Threads'", timeout=10)
        print(f"\nPID {pid}:")
        print(o2.read().decode().strip())
else:
    print("\nNO PROCESS — dead")

# GPU
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
print(f"\nGPU3: {o.read().decode().strip()}")

# tmux
s, o, e = c.exec_command("tmux ls 2>/dev/null", timeout=10)
print(f"tmux: {o.read().decode().strip()}")

c.close()
