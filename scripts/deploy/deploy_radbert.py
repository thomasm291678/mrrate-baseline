import paramiko, time

HOST = "10.176.60.72"
USER = "jiaqigu"
PASS = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)
print("Connected.")

client.exec_command("tmux kill-session -t radbert 2>/dev/null; pkill -9 -f test_radbert.py 2>/dev/null; sleep 1")
print("Old cleaned.")

sftp = client.open_sftp()
with open("C:/Users/HP/Documents/5555/test_radbert.py", "r", encoding="utf-8") as f:
    content = f.read()
with sftp.open(f"{REMOTE}/test_radbert.py", "w") as f:
    f.write(content)
sftp.close()
print("Script uploaded.")

shell_cmd = (
    "export HF_ENDPOINT=https://hf-mirror.com && "
    "source ~/hidnet_env/bin/activate 2>/dev/null || conda activate hidnet_env 2>/dev/null; "
    "python -u test_radbert.py 2>&1 | tee radbert_result.log"
)
cmd = f'cd {REMOTE} && tmux new-session -d -s radbert "{shell_cmd}"'
print("Starting:", cmd)
_, stdout, stderr = client.exec_command(cmd)
out = stdout.read().decode()
err = stderr.read().decode()
if err:
    print("STDERR:", err)
if out:
    print("STDOUT:", out)
print("Done.")
client.close()
