import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Find PIDs on GPU0
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null", timeout=10)
print("GPU processes:", o.read().decode().strip())

# Kill any process using GPU0 (bus ID 00000000:01:00.0 or similar)
s, o, e = c.exec_command(
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -20",
    timeout=10)
print("All GPU procs:", o.read().decode().strip())

# Check what process has GPU 0 memory
s, o, e = c.exec_command("nvidia-smi pmon -c 1 2>/dev/null", timeout=10)
print("PMON:", o.read().decode().strip()[:500])

c.close()
