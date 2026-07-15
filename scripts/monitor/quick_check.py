import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)
# Check GPU2 (new training) and GPU status
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader", timeout=10)
print("GPU:", stdout.read().decode().strip())
# Latest training log
stdin, stdout, stderr = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log 2>/dev/null | head -3", timeout=10)
print("Logs:", stdout.read().decode().strip())
# Last 10 training lines
stdin, stdout, stderr = c.exec_command(
    "tail -10 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log 2>/dev/null | head -1) 2>/dev/null",
    timeout=10)
print("Latest:", stdout.read().decode().strip())
c.close()
