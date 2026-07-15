import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_140341.log"

s, o, e = c.exec_command(f"tail -15 {log}", timeout=10)
print(o.read().decode())

# GPU
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("GPU3:", o.read().decode().strip())

# latest_step.pt
s, o, e = c.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>/dev/null || echo 'no latest_step.pt'",
    timeout=10)
print(o.read().decode().strip())

# Process alive
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep | wc -l", timeout=10)
print(f"Processes alive: {o.read().decode().strip()}")

c.close()
