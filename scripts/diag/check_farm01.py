import paramiko

FARM01 = {"hostname": "10.176.60.71", "username": "jiaqigu", "password": "lijia7272"}

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(**FARM01, timeout=30)

cmds = [
    'echo "=== GPU6 ==="',
    'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep "^6"',
    'echo "=== ENV ==="',
    'test -f /home/jiaqigu/hidnet_env/bin/python && echo ENV_OK || echo NO_ENV',
    'echo "=== MODEL ==="',
    'ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/best_model.pt 2>/dev/null || echo NO_MODEL',
    'echo "=== CODE ==="',
    'ls -lh /home/jiaqigu/mrrate_hidnet/src/ 2>/dev/null',
    'ls -lh /home/jiaqigu/mrrate_hidnet/scripts/ 2>/dev/null',
    'echo "=== PROCESSES ==="',
    'ps aux | grep -E "python|train" | grep jiaqigu | grep -v grep',
]

stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
print(stdout.read().decode(errors="replace"))
c.close()
