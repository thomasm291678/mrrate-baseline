import paramiko, time

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
    
    sftp = c.open_sftp()
    sftp.put(r"C:\Users\HP\Documents\5555\train.py", "/home/jiaqigu/mrrate_hidnet/scripts/train.py")
    sftp.put(r"C:\Users\HP\Documents\5555\encoder.py", "/home/jiaqigu/mrrate_hidnet/encoder.py")
    sftp.put(r"C:\Users\HP\Documents\5555\eval_report.py", "/home/jiaqigu/mrrate_hidnet/eval_report.py")
    sftp.put(r"C:\Users\HP\Documents\5555\unified_eval.py", "/home/jiaqigu/mrrate_hidnet/unified_eval.py")
    sftp.close()
    print(f"{name}: uploaded train.py + encoder.py + eval_report.py + unified_eval.py")
    c.close()

print("\nAll servers synchronized")
