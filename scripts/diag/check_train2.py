import paramiko
import time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

time.sleep(5)

cmds = [
    'echo "=== All jiaqigu procs ==="',
    'ps -u jiaqigu --no-headers 2>/dev/null',
    'echo "=== GPU6 ==="',
    'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep "^6"',
    'echo "=== Newest train log ==="',
    'ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1 | xargs tail -30',
    'echo "=== Watchdog log ==="',
    'cat /tmp/watchdog_out.log 2>/dev/null',
]

stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
out = stdout.read().decode(errors="replace")
print(out)

c.close()
