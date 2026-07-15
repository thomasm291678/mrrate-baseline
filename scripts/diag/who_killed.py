import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check dmesg for OOM kill around 16:47
s, o, e = c.exec_command(
    "dmesg -T 2>/dev/null | grep -iE 'killed|oom|out of memory' | tail -15",
    timeout=15)
print("=== dmesg OOM ===")
out = o.read().decode().strip()
print(out if out else "(none)")

# Check earlyoom log
s, o, e = c.exec_command(
    "journalctl -u earlyoom --no-pager -n 20 2>/dev/null || tail -50 /var/log/earlyoom.log 2>/dev/null || echo 'no earlyoom access'",
    timeout=15)
print("\n=== earlyoom ===")
out = o.read().decode().strip()
print(out)

# Check GPU monitor log
s, o, e = c.exec_command(
    "cat /var/log/gpu_monitor.log 2>/dev/null | tail -30; "
    "cat /mnt/nas1/disk07/root/nvmonitor/*.log 2>/dev/null | tail -30; "
    "echo '---end---'",
    timeout=15)
print("\n=== GPU monitor logs ===")
out = o.read().decode().strip()
print(out)

# Check if there's a resource limit
s, o, e = c.exec_command("ulimit -a | cat", timeout=10)
print("\n=== ulimit ===")
print(o.read().decode().strip())

c.close()
