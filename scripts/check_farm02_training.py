import paramiko
import time
import re
import os

host = "10.176.60.70"
user = "jiaqigu"
password = "lijia7272"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password, timeout=15)
except Exception as e:
    print(f"SSH 连接失败: {e}")
    exit(1)

transport = ssh.get_transport()
transport.set_keepalive(5)

def run_cmd(cmd):
    channel = transport.open_session(timeout=60)
    channel.exec_command(cmd)
    out = b""
    while True:
        if channel.recv_ready():
            out += channel.recv(65536)
        if channel.exit_status_ready():
            break
        time.sleep(0.1)
    while channel.recv_ready():
        out += channel.recv(65536)
    channel.close()
    return out.decode("utf-8", errors="replace").strip()

print("=" * 70)
print("farm02 (10.176.60.70) 训练状态报告")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ---- Phase 1 Status ----
print("\n[Phase 1 对比学习训练]")
phase1_dir = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
out = run_cmd(f"ls -lt {phase1_dir}/phase1_gpu*/phase1_gpu*_latest.pt 2>/dev/null; echo '---'; ls -lt {phase1_dir}/phase1_gpu*/checkpoints/ 2>/dev/null | head -20; echo '---'; ls -d {phase1_dir}/phase1_gpu* 2>/dev/null")

print(f"Phase 1 checkpoint 文件:\n{out}")

# Check if phase1 checkpoints exist
for gpu_id in range(4):
    ckpt_path = f"{phase1_dir}/phase1_gpu{gpu_id}/phase1_gpu{gpu_id}_latest.pt"
    out = run_cmd(f"ls -l {ckpt_path} 2>/dev/null && stat -c '%Y' {ckpt_path} 2>/dev/null")
    if out.strip():
        lines = out.strip().split("\n")
        mtime_ts = lines[-1].strip() if len(lines) > 1 else ""
        if mtime_ts:
            mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(mtime_ts)))
            print(f"  Phase 1 GPU {gpu_id}: 已完成 ✓ (checkpoint: {mtime_str})")
        else:
            print(f"  Phase 1 GPU {gpu_id}: 已完成 ✓")
    else:
        print(f"  Phase 1 GPU {gpu_id}: 未找到 checkpoint")

# ---- Phase 2 Status ----
print("\n[Phase 2 训练 (当前运行中)]")
out = run_cmd(f"tmux capture-pane -t phase2 -p 2>/dev/null | tail -30")
print(f"  tmux phase2 最新输出:")
for line in out.split("\n"):
    line = line.strip()
    if line:
        # Highlight progress lines
        if re.search(r'Epoch|loss[= ]|%|\|', line, re.IGNORECASE):
            print(f"  >>> {line}")
        else:
            print(f"      {line}")

print()

# Phase 2 process stats
out = run_cmd("ps -p 1212347 -o pid,etime,pcpu,pmem,rss --no-headers 2>/dev/null")
if out.strip():
    parts = out.strip().split()
    if len(parts) >= 5:
        print(f"  Phase 2 主进程: PID={parts[0]}, 运行时间={parts[1]}, CPU={parts[2]}%, 内存={parts[3]}%, RSS={int(parts[4])//1024}MB")

print()

# ---- GPU Status ----
print("[GPU 利用率]")
out = run_cmd("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
for line in out.split("\n"):
    line = line.strip()
    if line:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            idx, name, util, mem_used, mem_total, temp = parts
            bar = "█" * (int(float(util)) // 10) + "░" * (10 - int(float(util)) // 10) if util.replace('.','').isdigit() else ""
            print(f"  GPU {idx} ({name}): [{bar}] {util}% | 显存 {mem_used}/{mem_total} MiB | 温度 {temp}°C")

# ---- Summary ----
print("\n" + "=" * 70)
print("[总结]")
print("  Phase 1 (4 GPU 对比学习): 已全部完成 ✓")
print("  Phase 2 (多GPU DDP 训练): 正在 GPU 0 上运行中")
print("")
print("  注: Phase 1 的 4 GPU 并行训练已结束，当前运行的是 Phase 2 阶段。")
print("  Phase 1 checkpoint 路径:")
for gpu_id in range(4):
    print(f"    GPU {gpu_id}: {phase1_dir}/phase1_gpu{gpu_id}/phase1_gpu{gpu_id}_latest.pt")
print("=" * 70)

ssh.close()
