import paramiko
import os
import glob
from datetime import datetime

host = "10.154.32.115"
user = "jiaqigu"
password = "lijia7272"
local_dir = r"C:\Users\HP\Documents\5555\training_logs"

os.makedirs(local_dir, exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=15)

sftp = client.open_sftp()

remote_log_dir = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
stdin, stdout, stderr = client.exec_command(
    f"ls -t {remote_log_dir}/train_farm02_*.log 2>/dev/null | head -1"
)
latest_log = stdout.read().decode('utf-8').strip()

if latest_log:
    local_path = os.path.join(local_dir, os.path.basename(latest_log))
    try:
        sftp.get(latest_log, local_path)
        print(f"Downloaded: {os.path.basename(latest_log)} -> {local_path}")
    except Exception as e:
        print(f"Download error: {e}")

    stdin, stdout, stderr = client.exec_command(f"wc -l {latest_log}")
    line_count = stdout.read().decode('utf-8').strip()
    print(f"Log size: {line_count}")

sftp.close()
client.close()

local_logs = sorted(glob.glob(os.path.join(local_dir, "train_farm02_*.log")))
print(f"\nTotal local logs: {len(local_logs)}")
for log in local_logs:
    size = os.path.getsize(log)
    print(f"  {os.path.basename(log)} ({size:,} bytes)")
