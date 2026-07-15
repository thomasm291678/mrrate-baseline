import paramiko, time

time.sleep(180)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/fix_launch.log 2>/dev/null", timeout=10)
print(stdout.read().decode())

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
print("\nGPU:", stdout.read().decode().strip())

c.close()
