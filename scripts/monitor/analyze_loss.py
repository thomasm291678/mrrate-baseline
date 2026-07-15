import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)

# V3 current log
cmds = [
    'echo "=== V3最新20条loss ==="',
    'grep "loss=" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260712_223225.log 2>/dev/null | awk -F"loss=" "{print \$2}" | awk "{print \$1}" | tail -20',
    'echo "=== V3 loss统计 ==="',
    'grep "loss=" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260712_223225.log 2>/dev/null | awk -F"loss=" "{print \$2}" | awk "{print \$1}" | python3 -c "import sys; nums=[float(l.strip()) for l in sys.stdin if l.strip()]; print(f\"min={min(nums):.4f} max={max(nums):.4f} mean={sum(nums)/len(nums):.4f} count={len(nums)}\")" 2>/dev/null || echo "calc failed"',
    'echo "=== V1 高loss实例(V1训练第10个epoch) ==="',
    'grep -A0 "loss=" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_farm02_*.log 2>/dev/null | grep "E0\(09\|10\)" | awk -F"loss=" "{print \$2}" | awk "{print \$1}" | sort -n | head -30',
    'echo "=== V1 整体loss范围 ==="',
    'grep "loss=" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_farm02_*.log 2>/dev/null | awk -F"loss=" "{print \$2}" | awk "{print \$1}" | python3 -c "import sys; nums=[float(l.strip()) for l in sys.stdin if l.strip()]; print(f\"min={min(nums):.4f} max={max(nums):.4f} mean={sum(nums)/len(nums):.4f} count={len(nums)}\")" 2>/dev/null || echo "calc failed"',
]
stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
print(stdout.read().decode(errors="replace"))
c.close()
