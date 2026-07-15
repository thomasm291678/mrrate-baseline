import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)

_, o, _ = c.exec_command("ps aux | grep python | grep -v grep | head -5")
print("PROCS:", o.read().decode().strip() or "none")

_, o, _ = c.exec_command("ls -la /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2*.pt 2>&1")
print("CKPTS:", o.read().decode().strip()[:200] or "none")

_, o, _ = c.exec_command("tail -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log 2>&1")
print("LOG:", o.read().decode().strip() or "none")

_, o, _ = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -4")
print("GPU:\n" + o.read().decode().strip())

c.close()
