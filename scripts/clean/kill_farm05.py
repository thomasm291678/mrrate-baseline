import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Kill all GPU processes on farm05
c.exec_command("pkill -9 -f 'python.*train' 2>/dev/null; true", timeout=5)
c.exec_command("pkill -9 -f 'pt_data_worker' 2>/dev/null; true", timeout=5)
time.sleep(5)

# Also fuser-kill each GPU
for gpu in [0, 1, 2, 3]:
    c.exec_command(f"fuser -k /dev/nvidia{gpu} 2>/dev/null; true", timeout=5)
time.sleep(10)

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader", timeout=10)
print(o.read().decode().strip())

c.close()
print("Done")
