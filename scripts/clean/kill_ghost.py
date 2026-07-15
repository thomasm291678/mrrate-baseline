import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("kill -9 2595483 2>/dev/null; true", timeout=5)
time.sleep(5)

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3 after kill:", o.read().decode().strip())

s, o, e = c.exec_command("ps -p 2595483 2>/dev/null && echo ALIVE || echo DEAD", timeout=10)
print("PID 2595483:", o.read().decode().strip())

# If still dirty, try harder
s, o, e = c.exec_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep 2595483", timeout=10)
out = o.read().decode().strip()
if out:
    print(f"Still on GPU: {out}")
    c.exec_command("fuser -v /dev/nvidia3 2>/dev/null; true", timeout=5)
    c.exec_command("kill -9 $(fuser /dev/nvidia3 2>/dev/null) 2>/dev/null; true", timeout=5)
    time.sleep(5)
    s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
    print("GPU3 after harder kill:", o.read().decode().strip())

c.close()
