import paramiko, time

time.sleep(60)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_farm05_*.log 2>/dev/null | head -1",
    timeout=10)
log = o.read().decode().strip()

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader",
    timeout=10)
print(o.read().decode().strip())

c.close()
