import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# 1. Kill any running v4 process first
c.exec_command("pkill -9 -f 'train_v4' 2>/dev/null; true", timeout=5)
print("Killed any running train_v4")

# 2. /dev/shm 共享内存
s, o, e = c.exec_command("df -h /dev/shm", timeout=10)
print(f"\n=== /dev/shm ===")
print(o.read().decode().strip())

# 3. 共享内存总量
s, o, e = c.exec_command("ls -lh /dev/shm/ | head -20", timeout=10)
print(f"\n=== /dev/shm contents ===")
print(o.read().decode().strip())

# 4. 所有历史崩溃时间对比
s, o, e = c.exec_command(
    "grep -n ' S00' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_132326.log | tail -3",
    timeout=10)
print(f"\n=== Crash 1 (13:25) last steps ===")
print(o.read().decode().strip())

s, o, e = c.exec_command(
    "grep -n ' S00' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_20260713_140341.log | tail -3",
    timeout=10)
print(f"\n=== Crash 2 (14:04) last steps ===")
print(o.read().decode().strip())

# 5. 检查 latest_step.pt 是否正常
s, o, e = c.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt",
    timeout=10)
print(f"\n=== latest_step.pt ===")
print(o.read().decode().strip())

# 6. 检查磁盘空间
s, o, e = c.exec_command("df -h /home/jiaqigu/", timeout=10)
print(f"\n=== Disk /home/jiaqigu ===")
print(o.read().decode().strip())

# 7. 网络挂载 /mnt/nas1
s, o, e = c.exec_command(
    "df -h /mnt/nas1/disk07/public/mr_data/ 2>/dev/null; "
    "ls /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv 2>/dev/null && echo 'NAS OK' || echo 'NAS BROKEN'",
    timeout=15)
print(f"\n=== NAS mount ===")
print(o.read().decode().strip())

# 8. 用 strace 思路：check if any zombie python processes from previous runs
s, o, e = c.exec_command(
    "ps aux | grep -E 'train_v4|python.*mrrate' | grep -v grep",
    timeout=10)
print(f"\n=== Zombie processes ===")
out = o.read().decode().strip()
print(out if out else "(none)")

# 9. GPU error log
s, o, e = c.exec_command(
    "cat /var/log/syslog 2>/dev/null | grep -i 'kernel.*nvidia\|nvidia.*error\|xid' | tail -20; "
    "dmesg -T 2>/dev/null | grep -i nvidia | tail -20",
    timeout=15)
print(f"\n=== GPU kernel errors ===")
out = o.read().decode().strip()
print(out if out else "(none or no access)")

# 10. 检查是否在固定步数崩溃
s, o, e = c.exec_command(
    "for f in /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_*.log; do "
    "echo '---'; basename $f; "
    "grep -c ' S0' $f 2>/dev/null; "
    "tail -2 $f 2>/dev/null; "
    "done",
    timeout=15)
print(f"\n=== All V4 logs summary ===")
print(o.read().decode().strip())

c.close()
