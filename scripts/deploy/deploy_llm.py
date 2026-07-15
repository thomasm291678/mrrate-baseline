import paramiko

HOST = "10.176.60.72"; USER = "jiaqigu"; PASS = "lijia7272"; REMOTE = "/home/jiaqigu/mrrate_hidnet"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

client.exec_command("pkill -f eval_llm 2>/dev/null; tmux kill-session -t llmeval 2>/dev/null; sleep 1")

# Upload both files
sftp = client.open_sftp()
for fname in ["eval_llm.py", "save_prompt.py"]:
    with open(f"C:/Users/HP/Documents/5555/{fname}", "r", encoding="utf-8") as f:
        content = f.read()
    with sftp.open(f"{REMOTE}/{fname}", "w") as f:
        f.write(content)
sftp.close()
print("Files uploaded")

# Save prompt to NAS
_, o, _ = client.exec_command(f"cd {REMOTE} && source ~/hidnet_env/bin/activate && mkdir -p /mnt/nas1/disk07/public/jiaqigu/evaluation && python3 save_prompt.py")
print(o.read().decode().strip())

# Start eval
shell = "export HF_ENDPOINT=https://hf-mirror.com; source ~/hidnet_env/bin/activate; python -u eval_llm.py --max-calls 500 2>&1 | tee eval_llm_v2_result.log"
cmd = f'cd {REMOTE} && tmux new-session -d -s llmeval "{shell}"'
client.exec_command(cmd)
print("eval started")
client.close()
