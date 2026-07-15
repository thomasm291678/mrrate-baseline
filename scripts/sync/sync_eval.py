import paramiko, io

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Read local file
with open(r"C:\Users\HP\Documents\5555\evaluation_v2.py", "rb") as f:
    data = f.read()

sftp = c.open_sftp()
# Update git working copy
sftp.putfo(io.BytesIO(data), "/home/jiaqigu/mrrate_eval_git/evaluation/v2/evaluation_v2.py")
# Update NAS
sftp.putfo(io.BytesIO(data), "/mnt/nas1/disk07/public/jiaqigu/evaluation/v2/evaluation_v2.py")
sftp.close()

print("Files synced.")

# Git commit + push
cmds = [
    "cd /home/jiaqigu/mrrate_eval_git && git add evaluation/v2/evaluation_v2.py",
    'cd /home/jiaqigu/mrrate_eval_git && git commit -m "refactor: default 37-class only, 14-class moved to optional flag"',
    "cd /home/jiaqigu/mrrate_eval_git && git push origin main",
]
for cmd in cmds:
    s, o, e = c.exec_command(cmd, timeout=15)
    print(cmd.split("&&")[-1].strip())
    print(o.read().decode(), e.read().decode())

c.close()
print("Done.")
