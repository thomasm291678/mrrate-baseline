import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

print("Waiting 120s for first steps...")
time.sleep(120)

stdin, stdout, stderr = c.exec_command(
    "tail -15 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print(stdout.read().decode(errors="replace"))

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
print("GPU6:", stdout.read().decode().strip())

c.close()
