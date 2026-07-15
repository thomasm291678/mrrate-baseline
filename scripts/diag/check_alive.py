import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if process alive
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep | wc -l", timeout=10)
alive = int(o.read().decode().strip())
print(f"train_v4 processes alive: {alive}")

# Get latest log and compute time since last entry
log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_124845.log"
s, o, e = c.exec_command(f"tail -3 {log}", timeout=10)
last_lines = o.read().decode().strip()
print(f"Last 3 log lines:\n{last_lines}")

# Also check if there's a more recent log file
s, o, e = c.exec_command(
    "ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_*.log 2>/dev/null | head -3",
    timeout=10)
print(f"\nLog files:\n{o.read().decode().strip()}")

# Check GPU running processes
s, o, e = c.exec_command(
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null",
    timeout=10)
print(f"\nGPU processes:\n{o.read().decode().strip()}")

c.close()
