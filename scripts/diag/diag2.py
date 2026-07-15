import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Get latest log
stdin, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log | head -1", timeout=10)
log = o.read().decode().strip()
print(f"Log: {log}")

# Last 20 lines
stdin, o, e = c.exec_command(f"tail -20 {log}", timeout=10)
print(o.read().decode())

# Check checkpoints
stdin, o, e = c.exec_command(
    "ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/step_*.pt | head -5", timeout=10)
print("Checkpoints:", o.read().decode().strip())

c.close()
