import paramiko

HOST = "10.176.60.72"; USER = "jiaqigu"; PASS = "lijia7272"; REMOTE = "/home/jiaqigu/mrrate_hidnet"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

client.exec_command("pkill -f compare_llm 2>/dev/null; tmux kill-session -t compare 2>/dev/null; sleep 1")

sftp = client.open_sftp()
with open("C:/Users/HP/Documents/5555/compare_llm_vs_keyword.py", "r", encoding="utf-8") as f:
    content = f.read()
with sftp.open(f"{REMOTE}/compare_llm_vs_keyword.py", "w") as f:
    f.write(content)
sftp.close()

shell = "export HF_ENDPOINT=https://hf-mirror.com; source ~/hidnet_env/bin/activate; python -u compare_llm_vs_keyword.py 2>&1 | tee compare_result.log"
cmd = f'cd {REMOTE} && tmux new-session -d -s compare "{shell}"'
client.exec_command(cmd)
print("compare_llm_vs_keyword.py deployed and started in tmux 'compare'")
client.close()
