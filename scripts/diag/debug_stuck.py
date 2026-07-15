import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_220133.log"

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print("Log tail:")
print(o.read().decode())

s, o, e = c.exec_command(f"wc -l {log}", timeout=10)
print(f"Log lines: {o.read().decode().strip()}")

s, o, e = c.exec_command(f"stat -c '%Y' {log}", timeout=10)
mtime = int(o.read().decode().strip())
import time
now = time.time()
print(f"Last modified: {now - mtime:.0f}s ago")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Procs: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

# Check if process is stuck in I/O
s, o, e = c.exec_command("ps -u jiaqigu -o pid,stat,comm | grep 'train_v5' | head -5", timeout=10)
print("Process state:")
print(o.read().decode())

c.close()
