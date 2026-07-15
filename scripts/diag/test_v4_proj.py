import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Upload
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\encoder_v4.py", "/home/jiaqigu/mrrate_hidnet/encoder_v4.py")
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()
print("Uploaded")

s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && /home/jiaqigu/hidnet_env/bin/python encoder_v4.py",
    timeout=90)
print(o.read().decode())
c.close()
