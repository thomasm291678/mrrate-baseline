import paramiko, time
    
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=15)

# Create dirs
c.exec_command("mkdir -p /home/jiaqigu/mrrate_hidnet/scripts /home/jiaqigu/mrrate_hidnet/outputs/report_gen 2>/dev/null; true", timeout=5)

# Upload files
sftp = c.open_sftp()
files = [
    (r"C:\Users\HP\Documents\5555\train_v4.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py"),
    (r"C:\Users\HP\Documents\5555\encoder_v4.py", "/home/jiaqigu/mrrate_hidnet/encoder_v4.py"),
    (r"C:\Users\HP\Documents\5555\uniformer_blocks.py", "/home/jiaqigu/mrrate_hidnet/uniformer_blocks.py"),
    (r"C:\Users\HP\Documents\5555\server_code\mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py"),
]
for local, remote in files:
    try:
        sftp.put(local, remote)
        print(f"Uploaded: {local.split(chr(92))[-1]}")
    except Exception as ex:
        print(f"FAILED {local}: {ex}")
sftp.close()

# Verify
s, o, e = c.exec_command("ls /home/jiaqigu/mrrate_hidnet/scripts/train_v4.py && echo 'OK'", timeout=10)
print(o.read().decode().strip())

c.close()
