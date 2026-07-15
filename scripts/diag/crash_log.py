import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# GPU0 crash - last 30 lines before stop
s, o, e = c.exec_command("grep -i 'error\|traceback\|saved\|step_00' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260713_075047.log | tail -20", timeout=10)
print("GPU0 errors/saves:", o.read().decode())

# Full last 30 lines 
s, o, e = c.exec_command("tail -30 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260713_075047.log", timeout=10)
print("\nGPU0 last 30:", o.read().decode())

c.close()
