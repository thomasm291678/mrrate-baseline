import paramiko
import time

HOST = "10.176.60.70"
USER = "jiaqigu"
PASSWORD = "lijia7272"

def ssh_exec(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

print("=" * 70)
print("  Farm02 (10.176.60.70) 4-GPU Phase 1 训练状态检查")
print("=" * 70)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("[OK] SSH 连接成功\n")

    # 1. Check training progress for each GPU
    print("-" * 70)
    print("  Phase 1 对比学习训练进度")
    print("-" * 70)

    all_done = True
    for gpu_id in range(4):
        session = f"gpu{gpu_id}"
        cmd = f"tmux capture-pane -t {session} -p | grep -E 'loss=|E0' | tail -1"
        out, err = ssh_exec(ssh, cmd)

        if err:
            print(f"  GPU {gpu_id} ({session}): [ERROR] {err}")
            all_done = False
        elif out:
            print(f"  GPU {gpu_id} ({session}): {out}")
            all_done = False
        else:
            print(f"  GPU {gpu_id} ({session}): 无输出 (session 可能已完成或不存在)")
            # Check if session exists
            check_cmd = f"tmux has-session -t {session} 2>&1"
            check_out, _ = ssh_exec(ssh, check_cmd)
            if check_out:
                print(f"    -> session 不存在，任务可能已完成")

    # 2. Check if train_v5 process is still running
    print(f"\n{'-' * 70}")
    print("  train_v5 进程状态")
    print("-" * 70)
    proc_out, _ = ssh_exec(ssh, "ps aux | grep -v grep | grep train_v5")
    if proc_out:
        print(proc_out)
        all_done = False
    else:
        print("  无 train_v5 进程运行")

        # If all done, get finish time via tmux history
        print(f"\n{'-' * 70}")
        print("  训练完成时间 (查看 tmux 最近输出)")
        print("-" * 70)
        for gpu_id in range(4):
            session = f"gpu{gpu_id}"
            cmd = f"tmux capture-pane -t {session} -p -S -500 | tail -20"
            out, err = ssh_exec(ssh, cmd)
            if out:
                print(f"\n  GPU {gpu_id} ({session}) 末尾输出:")
                print("  " + out.replace("\n", "\n  "))
            else:
                print(f"  GPU {gpu_id} ({session}): 无数据或 session 不存在")

    # 3. nvidia-smi GPU utilization
    print(f"\n{'-' * 70}")
    print("  GPU 利用率 (nvidia-smi)")
    print("-" * 70)
    smi_out, _ = ssh_exec(ssh, "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
    if smi_out:
        for line in smi_out.split("\n"):
            print(f"  {line.strip()}")
    else:
        print("  无法获取 nvidia-smi 信息")

    # Summary
    print(f"\n{'=' * 70}")
    if all_done:
        print("  结论: 全部 4 GPU 训练已完成")
    else:
        print("  结论: 训练仍在进行中")
    print("=" * 70)

except Exception as e:
    print(f"[FAIL] SSH 连接或执行失败: {e}")
finally:
    ssh.close()
