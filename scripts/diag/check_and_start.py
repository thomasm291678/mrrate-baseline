import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if startup.log exists
stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/startup.log 2>/dev/null || echo 'no log'", timeout=10)
print("Startup:", stdout.read().decode()[:200])

# Check GPU1
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^1,'", timeout=10)
print("GPU1:", stdout.read().decode().strip())

# Check if eval_ train log exists
stdin, stdout, stderr = c.exec_command(
    "ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_*.log 2>/dev/null | head -1", timeout=10)
print("Eval log:", stdout.read().decode().strip()[:200])

# Check deps
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/python -c 'import evaluate; print(\"evaluate OK\")' 2>&1", timeout=10)
print("Deps:", stdout.read().decode().strip())

# Start directly if not running
start_cmd = (
    "cd /home/jiaqigu/mrrate_hidnet && "
    "nohup bash start_eval.sh > outputs/report_gen/startup.log 2>&1 &"
)
stdin, stdout, stderr = c.exec_command(start_cmd, timeout=10)
print("Triggered start_eval.sh")

c.close()
