import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check YOUR cron script
s, o, e = c.exec_command("cat /home/jiaqigu/mrrate_hidnet/check_monitor.sh 2>/dev/null || echo 'NOT FOUND'", timeout=10)
print("=== check_monitor.sh ===")
print(o.read().decode().strip())

# Check the GPU monitor daemon
s, o, e = c.exec_command("cat /mnt/nas1/disk07/root/nvmonitor/gpu_monitor.py 2>/dev/null | head -80", timeout=10)
print("\n=== gpu_monitor.py (first 80 lines) ===")
print(o.read().decode().strip())

# Check gpu_util_monitor service
s, o, e = c.exec_command("systemctl cat gpu_util_monitor.service 2>/dev/null || cat /etc/systemd/system/gpu_util_monitor.service 2>/dev/null || echo 'cannot read service file'", timeout=10)
print("\n=== gpu_util_monitor.service ===")
print(o.read().decode().strip())

# Check earlyoom
s, o, e = c.exec_command("systemctl cat earlyoom.service 2>/dev/null | head -20", timeout=10)
print("\n=== earlyoom.service ===")
print(o.read().decode().strip())

# Recent dmesg for any kill signals
s, o, e = c.exec_command("dmesg -T 2>/dev/null | grep -iE 'earlyoom|kill|sigterm|sigkill' | tail -20", timeout=15)
print("\n=== Recent kills ===")
out = o.read().decode().strip()
print(out if out else "(no recent kills in dmesg)")

# Check systemd journal for our process
s, o, e = c.exec_command("journalctl -u gpu_util_monitor --no-pager -n 30 2>/dev/null || echo 'no journal access'", timeout=10)
print("\n=== GPU monitor journal ===")
print(o.read().decode().strip())

c.close()
