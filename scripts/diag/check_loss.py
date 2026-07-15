import paramiko
import time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

print("Waiting 60s for model loading...")
time.sleep(60)

cmds = [
    'echo "=== GPU6 ==="',
    'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep "^6"',
    'echo "=== Training ==="',
    'tail -20 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)',
    'echo "=== Python PID ==="',
    'ps -u jiaqigu | grep python',
]

stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
print(stdout.read().decode(errors="replace"))

c.close()
