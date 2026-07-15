import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\train.py", "/home/jiaqigu/mrrate_hidnet/scripts/train.py")
sftp.put(r"C:\Users\HP\Documents\5555\eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py")
sftp.close()

print("Uploaded: train.py + eval_report.py")
print("Note: NOT restarting — code fixes apply on next run")
c.close()
