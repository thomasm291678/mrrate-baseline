import paramiko
import time

host = "10.176.60.70"
user = "jiaqigu"
password = "lijia7272"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(host, username=user, password=password, timeout=15)
    print(f"=== 已连接到 {host} ===\n")

    # 检查 train_v5 进程是否还在运行
    stdin, stdout, stderr = client.exec_command("ps aux | grep train_v5 | grep -v grep")
    train_procs = stdout.read().decode().strip()
    all_done = not bool(train_procs)

    print("=" * 70)
    print("Phase 1 对比学习训练状态 - 4 GPU 并行")
    print("=" * 70)

    for gpu_id in range(4):
        session = f"gpu{gpu_id}"
        cmd = f"tmux capture-pane -t {session} -p | grep -E 'loss=|E00' | tail -1"
        stdin, stdout, stderr = client.exec_command(cmd)
        output = stdout.read().decode().strip()
        err = stderr.read().decode().strip()

        if err:
            print(f"\n[GPU {gpu_id}] tmux session '{session}' 错误: {err}")
        elif output:
            print(f"\n[GPU {gpu_id}] 最新训练日志: {output}")
        else:
            print(f"\n[GPU {gpu_id}] tmux session '{session}' 无 loss/E00 输出")

    print("\n" + "=" * 70)
    print("GPU 利用率 (nvidia-smi)")
    print("=" * 70)

    stdin, stdout, stderr = client.exec_command(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
    )
    smi_output = stdout.read().decode().strip()
    if smi_output:
        print(f"\n{'GPU':<6} {'Name':<30} {'Util':<8} {'Memory':<20} {'Temp':<8}")
        print("-" * 75)
        for line in smi_output.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, util, mem_used, mem_total, temp = parts
                mem_str = f"{mem_used} / {mem_total}"
                print(f"GPU {idx:<2} {name:<30} {util:<8} {mem_str:<20} {temp:<8}")

    print("\n" + "=" * 70)

    if all_done:
        print("\n全部 GPU 训练已完成！无 train_v5 进程运行中。")

        stdin, stdout, stderr = client.exec_command(
            "ls -lt /home/jiaqigu/*.log 2>/dev/null | head -1 | awk '{print $6, $7, $8, $9}'"
        )
        log_info = stdout.read().decode().strip()
        if log_info:
            print(f"最近修改的日志文件: {log_info}")

        for gpu_id in range(4):
            cmd = f"tmux capture-pane -t gpu{gpu_id} -p | grep -i 'finished\|done\|complete\|cost\|time' | tail -3"
            stdin, stdout, stderr = client.exec_command(cmd)
            tail_out = stdout.read().decode().strip()
            if tail_out:
                print(f"\n[GPU {gpu_id}] 完成信息:")
                print(tail_out)
    else:
        print(f"\n检测到 train_v5 进程仍在运行:")
        for line in train_procs.split("\n"):
            print(f"  {line}")

except Exception as e:
    print(f"连接错误: {e}")
finally:
    client.close()
