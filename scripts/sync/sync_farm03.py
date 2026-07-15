import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=15)

# Sync from farm04 using scp
s, o, e = c.exec_command(
    "scp -o StrictHostKeyChecking=no -r 10.176.60.71:/home/jiaqigu/mrrate_hidnet/ /home/jiaqigu/ 2>&1",
    timeout=120)
print("scp output:", o.read().decode()[-500:] if o else "none")

# Check if synced
s, o, e = c.exec_command("ls /home/jiaqigu/mrrate_hidnet/scripts/train_v4.py 2>/dev/null && echo 'OK' || echo 'FAIL'", timeout=10)
print(o.read().decode().strip())

c.close()
