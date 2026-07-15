import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.154.32.115", username="jiaqigu", password="lijia7272", timeout=20)

cmds = [
    'echo "=== farm02 checkpoints ==="',
    'ls -lhR /home/jiaqigu/mrrate_hidnet/checkpoints/ 2>/dev/null || echo "no dir"',
    'echo "=== farm02 outputs (pt files) ==="',
    'find /home/jiaqigu/mrrate_hidnet/outputs/ -name "*.pt" 2>/dev/null | head -20',
    'echo "=== NAS checkpoints ==="',
    'ls -lhR /mnt/nas1/disk07/public/qi/checkpoints/ 2>/dev/null || echo "no dir"',
]
stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=20)
print(stdout.read().decode(errors="replace"))
c.close()

# Also check farm01
c2 = paramiko.SSHClient()
c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c2.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

cmds2 = [
    'echo "=== farm01 checkpoints ==="',
    'ls -lhR /home/jiaqigu/mrrate_hidnet/checkpoints/ 2>/dev/null || echo "no dir"',
    'echo "=== farm01 outputs (pt files) ==="',
    'ls -lhR /home/jiaqigu/mrrate_hidnet/outputs/report_gen/*.pt 2>/dev/null || echo "none"',
    'echo "=== save_interval ==="',
    'grep save_interval /home/jiaqigu/mrrate_hidnet/scripts/train.py | head -5',
]
stdin2, stdout2, stderr2 = c2.exec_command("; ".join(cmds2), timeout=20)
print(stdout2.read().decode(errors="replace"))
c2.close()
