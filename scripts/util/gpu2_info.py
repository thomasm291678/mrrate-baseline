import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check GPU2 process
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -oE '^[0-9]+'", timeout=10)
pids = o.read().decode().strip().split()
print(f"GPU PIDs: {pids}")

# Check GPU2 owner
for pid in pids:
    s, o, e = c.exec_command(f"ps -p {pid} -o pid,user,comm --no-headers 2>/dev/null", timeout=5)
    out = o.read().decode().strip()
    if out:
        print(out)

# Check if GPU2 (3301947) is killable by us
pid_gpu2 = "3301947"
s, o, e = c.exec_command(f"ps -p {pid_gpu2} -o pid,user,stat,etime --no-headers 2>/dev/null", timeout=5)
print(f"\nGPU2 PID {pid_gpu2}: {o.read().decode().strip()}")

# GPU mapping: find which bus ID = GPU2
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | grep '^2,'",
    timeout=10)
bus_gpu2 = o.read().decode().strip().split(",")[1].strip()
print(f"GPU2 bus_id: {bus_gpu2}")

# Match process to GPU2
s, o, e = c.exec_command(
    f"nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader | grep '{bus_gpu2}'",
    timeout=10)
print(f"GPU2 process: {o.read().decode().strip()}")

c.close()
