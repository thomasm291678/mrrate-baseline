import paramiko
import time

hostname = "10.176.60.70"
username = "jiaqigu"
password = "lijia7272"

def run_ssh_command(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

print("=" * 70)
print("  连接 farm02 (10.176.60.70) ...")
print("=" * 70)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(hostname=hostname, username=username, password=password, timeout=15)
    print("SSH 连接成功!\n")

    # 1. 检查 train_v5 进程是否存在
    print(">>> 检查 train_v5 进程:")
    out, err = run_ssh_command(ssh, "ps aux | grep -v grep | grep train_v5 || echo 'NO_PROCESS'")
    print(out if out else "无 train_v5 进程")
    has_process = "NO_PROCESS" not in out
    print()

    # 2. 获取每个 GPU tmux session 的最新进度
    print(">>> 各 GPU 训练进度:\n")
    gpu_results = []
    for gpu_id in range(4):
        session = f"gpu{gpu_id}"
        cmd = f"tmux capture-pane -t {session} -p 2>/dev/null | grep -E 'loss=|E0[0-9]+' | tail -3"
        out, err = run_ssh_command(ssh, cmd, timeout=10)
        gpu_results.append((gpu_id, out, err))

    for gpu_id, out, err in gpu_results:
        print(f"--- GPU {gpu_id} (tmux session gpu{gpu_id}) ---")
        if err and "can't find session" in err.lower():
            print("  [错误] tmux session 不存在")
        elif out:
            for line in out.split("\n"):
                print(f"  {line.strip()}")
        else:
            print("  [无输出] 可能是空 session 或无匹配内容")
        print()

    # 3. nvidia-smi GPU 利用率
    print(">>> NVIDIA-SMI GPU 利用率:\n")
    nvidia_cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
    out, err = run_ssh_command(ssh, nvidia_cmd, timeout=15)
    if out:
        print(f"  {'GPU':<6} {'Name':<28} {'Util':<8} {'Mem Used':<14} {'Temp':<6}")
        print(f"  {'-'*6} {'-'*28} {'-'*8} {'-'*14} {'-'*6}")
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                print(f"  {parts[0]:<6} {parts[1]:<28} {parts[2]:<8} {parts[3]:<14} {parts[4]:<6}")
    else:
        print("  nvidia-smi 无输出或执行失败")
        if err:
            print(f"  错误: {err}")
    print()

    # 4. 如果有进程，获取更详细的 epoch 信息
    if has_process:
        print(">>> 尝试从 tmux pane 获取更精确的 epoch 信息:\n")
        for gpu_id in range(4):
            session = f"gpu{gpu_id}"
            # Try broader grep to catch more progress lines
            cmd = f"tmux capture-pane -t {session} -p 2>/dev/null | tail -5"
            out, err = run_ssh_command(ssh, cmd, timeout=10)
            if out:
                print(f"  GPU {gpu_id} 最近 5 行:")
                for line in out.split("\n"):
                    print(f"    {line.strip()}")
                print()

    ssh.close()
    print("=" * 70)
    print("  检查完成")

except Exception as e:
    print(f"连接失败: {e}")
