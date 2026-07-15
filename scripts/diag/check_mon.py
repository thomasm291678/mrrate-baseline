import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/monitor.log")
mon_log = stdout.read().decode().strip()
print("=== Monitor Log ===")
print(mon_log)
print()

stdin, stdout, stderr = c.exec_command(
    "tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log|head -1)")
print("=== Training ===")
print(stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader|grep '^6'")
print("GPU6:", stdout.read().decode().strip())

c.close()
