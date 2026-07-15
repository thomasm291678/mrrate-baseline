import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

cmds = [
    'echo "=== MONITOR LOG (last 5) ==="',
    'tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/monitor.log',
    'echo "=== TRAINING (last 3) ==="',
    'tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log|head -1)',
    'echo "=== GPU6 ==="',
    'nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader|grep "^6"',
    'echo "=== PROC ==="',
    'ps -u jiaqigu|grep train.py|grep -v grep|wc -l',
]
stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=20)
print(stdout.read().decode(errors="replace"))
c.close()
