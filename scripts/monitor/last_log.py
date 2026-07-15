import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)
s,o,e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log | sed -n '1p'",
    timeout=10)
log = o.read().decode().strip()
print("Log:", log)
s,o,e = c.exec_command(f"tail -20 {log}", timeout=10)
print(o.read().decode())
c.close()
