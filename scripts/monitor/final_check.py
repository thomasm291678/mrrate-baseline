import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check process state
s, o, e = c.exec_command("cat /proc/3895976/status 2>/dev/null | grep -E 'State|Threads|VmRSS|VmSize'", timeout=10)
print("=== Process 3895976 ===")
print(o.read().decode().strip())

# Check latest log directly (not through tmux)
s, o, e = c.exec_command(
    "tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_163707.log 2>/dev/null",
    timeout=10)
print("\n=== Log tail ===")
print(o.read().decode().strip())

# Check if log is still being written
s, o, e = c.exec_command(
    "ls -la /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_163707.log",
    timeout=10)
print("\n=== Log file ===")
print(o.read().decode().strip())

# Check nvidia-smi for this process
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep 3895976", timeout=10)
print(f"\nGPU process: {o.read().decode().strip()}")

c.close()
