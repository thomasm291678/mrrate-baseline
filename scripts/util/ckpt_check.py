import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/ | head -10", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command("wc -l /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_204642.log", timeout=10)
print("Log lines:", o.read().decode().strip())

# Check if latest_step.pt still exists
s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>/dev/null", timeout=10)
print("latest_step.pt:", o.read().decode().strip())

c.close()
