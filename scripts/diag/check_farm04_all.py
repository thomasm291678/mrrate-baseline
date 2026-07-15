import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader",
    timeout=10)
print(o.read().decode())

# Check which processes are on which GPU
s, o, e = c.exec_command(
    "nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader",
    timeout=10)
print("\nProcesses:", o.read().decode().strip())

c.close()
