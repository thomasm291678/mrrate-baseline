import paramiko, io

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# read local README
with open(r"C:\Users\HP\Documents\5555\evaluation\README.md", "r", encoding="utf-8") as f:
    readme = f.read()

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(readme.encode()), "/home/jiaqigu/mrrate_eval_git/evaluation/README.md")
sftp.close()

c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git add evaluation/README.md")
c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git commit -m \"docs(evaluation): explain JSS fusion and expected effects\"")
s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git push origin main 2>&1")
print(o.read().decode(), e.read().decode())
print("Done: https://github.com/thomasm291678/mrrate-baseline/tree/main/evaluation")

c.close()
