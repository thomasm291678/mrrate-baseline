import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

cmd1 = "tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log 2>/dev/null | head -1)"
stdin, stdout, stderr = c.exec_command(cmd1, timeout=15)
log_lines = stdout.read().decode(errors="replace").strip()
print("=== Training Log ===")
print(log_lines if log_lines else "(empty / no log found)")

cmd2 = "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | grep '^6'"
stdin, stdout, stderr = c.exec_command(cmd2, timeout=15)
gpu = stdout.read().decode(errors="replace").strip()
print("=== GPU6 ===")
print(gpu if gpu else "(GPU6 not found or nvidia-smi failed)")

cmd3 = "ps -u jiaqigu | grep python | grep train | grep -v grep | wc -l"
stdin, stdout, stderr = c.exec_command(cmd3, timeout=15)
count = stdout.read().decode().strip()
print("=== Training Procs ===")
print(f"Active training processes: {count}")

c.close()
