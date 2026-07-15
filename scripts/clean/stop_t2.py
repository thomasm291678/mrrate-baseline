import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

c.exec_command("pkill -9 -f 'train_v4.*t2' 2>/dev/null; true", timeout=5)
print("Killed T2")

s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep", timeout=10)
print(o.read().decode().strip() or "(no processes)")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^3,|^7,'", timeout=10)
print(o.read().decode())

c.close()
