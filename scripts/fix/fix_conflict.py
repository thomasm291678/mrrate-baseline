import paramiko, io

GITIGNORE_CONTENT = """__pycache__/
*.pyc
.venv/
.trae/
.monitor_v5_state.json
*.pt
*.pth
*.safetensors
*.bin
*.ckpt
outputs/
env_package/
env_pkg/
hidnet_env_bundle.tar.gz
*.tar.gz
*.zip
*.tar
*.egg-info/
dist/
build/
.env
*.log
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(GITIGNORE_CONTENT.encode()), "/home/jiaqigu/mrrate_eval_git/.gitignore")
sftp.close()

c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git checkout --theirs README.md")
c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git add .gitignore README.md")

s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git -c core.editor=true rebase --continue 2>&1")
print("Rebase:", o.read().decode(), e.read().decode())

s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git push origin main 2>&1")
print("Push:", o.read().decode(), e.read().decode())

s, o, e = c.exec_command("cd /home/jiaqigu/mrrate_eval_git && git log --oneline -5")
print("Final log:", o.read().decode())

c.close()
