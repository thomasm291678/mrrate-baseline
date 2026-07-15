import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("fuser -k /dev/nvidia7 2>/dev/null; true", timeout=5)
print("GPU7 killed")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^7,'", timeout=10)
print("GPU7 after:", o.read().decode().strip())

c.close()
