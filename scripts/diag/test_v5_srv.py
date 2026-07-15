import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
sftp.close()

s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "/home/jiaqigu/hidnet_env/bin/python encoder_v5.py 2>&1",
    timeout=60)
print(o.read().decode().strip())
print(e.read().decode().strip())

c.close()
