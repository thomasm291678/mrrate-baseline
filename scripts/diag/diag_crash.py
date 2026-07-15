import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check dmesg for OOM killer
s, o, e = c.exec_command("dmesg -T | grep -i 'killed process\|out of memory\|oom' | tail -20 2>/dev/null || echo 'no dmesg access'", timeout=10)
print("=== dmesg OOM ===")
print(o.read().decode().strip() or "(no oom messages)")

# System memory
s, o, e = c.exec_command("free -h", timeout=10)
print("\n=== System RAM ===")
print(o.read().decode().strip())

# GPU processes on GPU3
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | grep -v 'No running'", timeout=10)
print("\n=== GPU Processes ===")
print(o.read().decode().strip())

# Check if v4 is currently running
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep", timeout=10)
out = o.read().decode().strip()
print(f"\n=== train_v4 ===")
print(out if out else "NOT RUNNING")

# Any core dumps
s, o, e = c.exec_command("ls -lh /home/jiaqigu/core* /home/jiaqigu/mrrate_hidnet/core* 2>/dev/null || echo 'no core dumps'", timeout=10)
print("\n=== Core dumps ===")
print(o.read().decode().strip())

# Kernel limits
s, o, e = c.exec_command("ulimit -a 2>/dev/null | head -15", timeout=10)
print("\n=== ulimit ===")
print(o.read().decode().strip())

c.close()
