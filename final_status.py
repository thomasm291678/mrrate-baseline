import paramiko, time

# farm04 GPU6
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu6_*.log | head -1)",
    timeout=10)
print("farm04 GPU6:", o.read().decode().strip())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^6,'",
    timeout=10)
print("GPU6:", o.read().decode().strip())

c.close()

# farm05 GPU0
time.sleep(5)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_farm05_*.log | head -1)",
    timeout=10)
print("\nfarm05 GPU0:", o.read().decode().strip())

s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^0,'",
    timeout=10)
print("GPU0:", o.read().decode().strip())

c.close()
