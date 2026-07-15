import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

commands = [
    "cd /home/jiaqigu && rm -rf mrrate_eval_git && mkdir mrrate_eval_git && cd mrrate_eval_git && git init",
    "cd /home/jiaqigu/mrrate_eval_git && git config user.email 'thomasm291678@users.noreply.github.com' && git config user.name 'thomasm291678'",
]

print("=== Init git on farm05 ===")
for cmd in commands:
    s, o, e = c.exec_command(cmd)
    time.sleep(1)
    print(o.read().decode(), e.read().decode())

print("\n=== Upload evaluation files to farm05 ===")
sftp = c.open_sftp()
import os
local_eval = r"C:\Users\HP\Documents\5555\evaluation"
for root, dirs, files in os.walk(local_eval):
    for fname in files:
        local_path = os.path.join(root, fname)
        rel_path = os.path.relpath(local_path, local_eval).replace("\\", "/")
        remote_dir = f"/home/jiaqigu/mrrate_eval_git/evaluation/{os.path.dirname(rel_path)}"
        c.exec_command(f"mkdir -p '{remote_dir}'")
        remote_path = f"/home/jiaqigu/mrrate_eval_git/evaluation/{rel_path}"
        sftp.put(local_path, remote_path)
        print(f"  Uploaded: {rel_path}")

# upload .gitignore and root evaluation_v2.py
sftp.put(r"C:\Users\HP\Documents\5555\.gitignore", "/home/jiaqigu/mrrate_eval_git/.gitignore")
sftp.put(r"C:\Users\HP\Documents\5555\evaluation\README.md", "/home/jiaqigu/mrrate_eval_git/README.md")
sftp.close()
print("  Done.\n")

print("=== Git add + commit + push ===")
cmds = [
    "cd /home/jiaqigu/mrrate_eval_git && git add .",
    "cd /home/jiaqigu/mrrate_eval_git && git commit -m 'evaluation: merge JSS evaluation.v1 (37-label) into unified evaluation_v2.py'",
    "cd /home/jiaqigu/mrrate_eval_git && git remote add origin https://thomasm291678:UnT2xNKG@github.com/thomasm291678/mrrate-baseline.git",
    "cd /home/jiaqigu/mrrate_eval_git && git branch -M main",
    "cd /home/jiaqigu/mrrate_eval_git && git push -u origin main 2>&1",
]
for cmd in cmds:
    s, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode())
    print(e.read().decode())

c.close()
