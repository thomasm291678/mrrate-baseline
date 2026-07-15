import paramiko

HOST = "10.176.60.70"
USER = "jiaqigu"
PASSWORD = "lijia7272"
GPUS = ["gpu0", "gpu1", "gpu2", "gpu3"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15)
print("SSH 连接成功\n")

# 1. 检查 tmux session 是否存在
print("=" * 60)
print("  tmux session 列表")
print("=" * 60)
stdin, stdout, stderr = ssh.exec_command("tmux list-sessions 2>&1")
out = stdout.read().decode("utf-8", errors="replace").strip()
print(out if out else "(无 tmux sessions)")

# 2. 获取每个 GPU tmux session 的最后 20 行（不用 grep 过滤）
print("\n" + "=" * 60)
print("  tmux 各 session 最后输出")
print("=" * 60)
for gpu in GPUS:
    cmd = f"tmux capture-pane -t {gpu} -p -S -20 2>&1"
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    if out:
        print(f"\n--- {gpu} ---")
        for line in out.split('\n'):
            print(f"  {line[:200]}")
    else:
        cmd2 = f"tmux capture-pane -t {gpu} -p 2>&1 | tail -20"
        stdin2, stdout2, stderr2 = ssh.exec_command(cmd2, timeout=30)
        out2 = stdout2.read().decode("utf-8", errors="replace").strip()
        if out2:
            print(f"\n--- {gpu} ---")
            for line in out2.split('\n'):
                print(f"  {line[:200]}")
        else:
            print(f"\n--- {gpu}: (空/不存在)")

# 3. 检查是否有模型 checkpoint 或日志文件
print("\n" + "=" * 60)
print("  最近修改的 checkpoint / 日志文件")
print("=" * 60)
stdin, stdout, stderr = ssh.exec_command("find /home/jiaqigu -name '*.pth' -o -name '*.pt' -o -name '*.tar' 2>/dev/null | head -20", timeout=30)
out = stdout.read().decode("utf-8", errors="replace").strip()
if out:
    print(out)

stdin, stdout, stderr = ssh.exec_command("ls -lt /home/jiaqigu/*.log 2>/dev/null | head -5", timeout=30)
out = stdout.read().decode("utf-8", errors="replace").strip()
if out:
    print("\n日志文件:")
    print(out)

# 也搜索 train_v5 常见输出目录
stdin, stdout, stderr = ssh.exec_command("find /home/jiaqigu -maxdepth 4 -name '*.log' -newer /home/jiaqigu -mtime -3 2>/dev/null | head -10", timeout=30)
out = stdout.read().decode("utf-8", errors="replace").strip()
if out:
    print("\n最近3天的日志:")
    print(out)

ssh.close()
print("\n检查完成")
