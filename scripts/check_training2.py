import paramiko

hostname = "10.176.60.70"
username = "jiaqigu"
password = "lijia7272"

def run_ssh_command(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(hostname=hostname, username=username, password=password, timeout=15)
    print("SSH 连接成功!\n")

    # 1. Check Phase 1 output directories for checkpoint files and timestamps
    print("=" * 70)
    print(">>> Phase 1 输出目录检查:")
    print("=" * 70)
    for gpu_id in range(4):
        phase1_dir = f"/home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase1_gpu{gpu_id}"
        cmd = f"ls -lt {phase1_dir}/ 2>/dev/null | head -10"
        out, err = run_ssh_command(ssh, cmd, timeout=10)
        print(f"\n--- phase1_gpu{gpu_id} ---")
        if out:
            print(out)
        else:
            print("  目录不存在或为空")

    # 2. Check if any train_v5 (phase 1) process is still running
    print("\n" + "=" * 70)
    print(">>> 检查 train_v5 Phase 1 进程:")
    print("=" * 70)
    cmd = "ps aux | grep -v grep | grep 'train_v5' | grep -v phase2"
    out, err = run_ssh_command(ssh, cmd, timeout=10)
    if out:
        print(out)
    else:
        print("无 train_v5 Phase 1 进程 (Phase 1 已结束)")

    # 3. Check phase2 progress from its tmux session
    print("\n" + "=" * 70)
    print(">>> Phase 2 训练进度 (tmux session phase2):")
    print("=" * 70)
    cmd = "tmux capture-pane -t phase2 -p 2>/dev/null | grep -E 'loss=|E0[0-9]+|epoch|Epoch' | tail -5"
    out, err = run_ssh_command(ssh, cmd, timeout=10)
    if out:
        print(out)
    else:
        print("无匹配输出，尝试获取最近行...")
        cmd2 = "tmux capture-pane -t phase2 -p 2>/dev/null | tail -10"
        out2, _ = run_ssh_command(ssh, cmd2, timeout=10)
        if out2:
            print(out2)
        else:
            print("tmux session phase2 无可用输出")

    # 4. Check Phase 1 log files for completion info
    print("\n" + "=" * 70)
    print(">>> Phase 1 完成状态 (检查日志):")
    print("=" * 70)
    for gpu_id in range(4):
        cmd = f"grep -i 'finished\|completed\|done\|saved.*latest\|End of training' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase1_gpu{gpu_id}/log.txt 2>/dev/null | tail -3"
        out, err = run_ssh_command(ssh, cmd, timeout=10)
        if out:
            print(f"\nGPU {gpu_id}:")
            print(out)

    # 5. Check ckpt file timestamps as completion time proxy
    print("\n" + "=" * 70)
    print(">>> Phase 1 checkpoint 文件时间戳 (用于确定完成时间):")
    print("=" * 70)
    for gpu_id in range(4):
        cmd = f"stat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase1_gpu{gpu_id}/phase1_gpu{gpu_id}_latest.pt 2>/dev/null"
        out, err = run_ssh_command(ssh, cmd, timeout=10)
        if out:
            print(f"\nGPU {gpu_id}:")
            for line in out.split("\n"):
                if "Modify" in line or "Birth" in line:
                    print(f"  {line.strip()}")

    ssh.close()

except Exception as e:
    print(f"连接失败: {e}")
