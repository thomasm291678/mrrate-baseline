import paramiko
import time
import re
import sys

host = "10.176.60.70"
user = "jiaqigu"
password = "lijia7272"

print(f"[*] 正在连接 {host} ...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(host, username=user, password=password, timeout=15)
except Exception as e:
    print(f"[!] SSH 连接失败: {e}")
    exit(1)

transport = ssh.get_transport()
transport.set_keepalive(5)

def run_cmd(cmd, timeout=30):
    channel = transport.open_session(timeout=timeout)
    channel.exec_command(cmd)
    out = b""
    while True:
        if channel.recv_ready():
            out += channel.recv(65536)
        if channel.exit_status_ready():
            break
        time.sleep(0.05)
    while channel.recv_ready():
        out += channel.recv(65536)
    channel.close()
    return out.decode("utf-8", errors="replace").strip()

print("[+] 已连接\n")
print("=" * 75)
print("  farm02 (10.176.60.70) — Phase 1 对比学习训练状态报告")
print(f"  检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

def parse_progress_line(line):
    results = {}

    epoch_match = re.search(r'Epoch\s*[:\s]*(\d+)', line)
    if epoch_match:
        results['epoch'] = int(epoch_match.group(1))

    loss_match = re.search(r'loss[=:\s]*([\d]+\.[\d]+)', line, re.IGNORECASE)
    if loss_match:
        results['loss'] = float(loss_match.group(1))

    pct_match = re.search(r'(\d+)%', line)
    if pct_match:
        results['progress'] = int(pct_match.group(1))
    else:
        batch_match = re.search(r'(\d+)\s*/\s*(\d+)', line)
        if batch_match:
            done, total = int(batch_match.group(1)), int(batch_match.group(2))
            if total > 0:
                results['progress'] = int(done / total * 100)
                results['batch_done'] = done
                results['batch_total'] = total

        iter_match = re.search(r'(\d+)/(\d+)', line)
        if iter_match and 'progress' not in results:
            done, total = int(iter_match.group(1)), int(iter_match.group(2))
            if total > 0 and total < 100000:
                results['