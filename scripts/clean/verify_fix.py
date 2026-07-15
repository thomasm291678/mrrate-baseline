import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Check if fix is on farm01
stdin, stdout, stderr = c.exec_command(
    "grep -n 'Resizing.*ckpt.*tokenizer' /home/jiaqigu/mrrate_hidnet/scripts/train.py 2>/dev/null && echo FIXED || echo NOT_FIXED")
print("Fix check:", stdout.read().decode().strip())

# Check for new log files
stdin, stdout, stderr = c.exec_command(
    "ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -5")
print("\nLog files:", stdout.read().decode().strip())

# Check GPU and processes
stdin, stdout, stderr = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
print("\nGPU6:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command("ps -u jiaqigu | grep python")
print("Python:", stdout.read().decode().strip())

c.close()
