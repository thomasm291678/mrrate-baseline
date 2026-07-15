import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Full log tail for crash reason
s, o, e = c.exec_command(
    "tail -30 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_124845.log",
    timeout=10)
print(o.read().decode())

# Kill GPU3 zombie
c.exec_command("fuser -k /dev/nvidia3 2>/dev/null; true", timeout=5)
print("\nGPU3 fuser-killed")

c.close()
