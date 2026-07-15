import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
print("GPU:", o.read().decode().strip())

s, o, e = c.exec_command("ps -u jiaqigu | grep train.py | grep -v grep", timeout=10)
procs = o.read().decode().strip()
print("Procs:", procs if procs else "NONE")

s, o, e = c.exec_command("ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log 2>/dev/null | head -4", timeout=10)
logs = o.read().decode().strip()
print("Recent logs:", logs)

c.close()
